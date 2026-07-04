"""Streaming-VLM patch for FlashDrive.

`StreamingMixin` caches the KV of earlier frames across windows so each new window
only prefills its latest frames. `create_streaming_attention_mask_sdpa` builds the
block attention mask matching that cached layout, and `convert_to_streaming_window`
converts a full 16-frame window to the 4-frame streaming form used by windows 1+.
"""

from typing import NamedTuple

import torch


class _QueryRegion(NamedTuple):
    """One block of query rows: a view's last frame, or the trailing traj+text."""

    q_start: int
    q_end: int
    num_attended_views: int  # views this region attends fully: view_ranges[:n]
    is_traj_text: bool
    kv_start: int  # KV start of the region's own (causal) block


def create_streaming_attention_mask_sdpa(
    frame_ranges: list[list[tuple[int, int]]],
    traj_text_range: tuple[int, int],
    kv_length: int,
    valid_length: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    extended_vision_ranges: bool,
) -> torch.Tensor:
    """Create the 4D streaming-VLM attention mask for SDPA (batch dim broadcasts).

    ``frame_ranges`` holds each view's per-frame ``[VS]..[VE]`` spans (end-exclusive);
    ``traj_text_range`` is everything after the last vision block.

    Query structure: [V0_F3] + [V1_F3] + ... + [Traj+Text]
    KV structure:    [System] + per-view blocks + [Traj+Text]

    Each view's last frame (F3) attends to the system prompt, all earlier views,
    its own earlier frames, and causally within its own F3. Traj+Text attends to
    every view plus causally within itself.

    ``extended_vision_ranges`` selects the per-view KV range:
    - False (v1, Qwen3-VL): tight ranges covering only the frame vision blocks.
    - True (v1.5, Cosmos): ranges extend back to the previous view's end, so the
      camera-name / frame-label tokens sitting in the gaps are attended too.

    Visual representation (■ = attend, □ = masked, ◣ = causal):

                  KV: | Sys | V0_block  | V1_block  | V2_block  | V3_block  | Traj+Text |
    Query:           |     | F0 F1 F2 F3| F0 F1 F2 F3| F0 F1 F2 F3| F0 F1 F2 F3|           |
    -----------------|-----|-----------|-----------|-----------|-----------|-----------|
    V0_F3            |  ■  |  ■  ■  ■ ◣ |  □  □  □  □|  □  □  □  □|  □  □  □  □|     □     |
    V3_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■ ◣ |     □     |
    Traj+Text        |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|     ◣     |
    """
    num_views = len(frame_ranges)

    # Per-view KV spans: extended pulls the start back to the prior view's end
    # (capturing camera/frame-label tokens); tight uses the view's own frames.
    view_ranges = []
    for view_idx in range(num_views):
        if extended_vision_ranges:
            view_start = (
                frame_ranges[0][0][0] if view_idx == 0 else frame_ranges[view_idx - 1][-1][1]
            )
        else:
            view_start = frame_ranges[view_idx][0][0]
        view_ranges.append((view_start, frame_ranges[view_idx][-1][1]))

    system_end = view_ranges[0][0]

    # Query regions: each view's last frame, then Traj+Text. Every region carries the
    # KV start of its own block (for the causal part) and how many preceding views it
    # attends (Traj+Text sits after every view, so it attends them all).
    q_offset = 0
    query_regions = []
    for view_idx in range(num_views):
        last_frame_start, last_frame_end = frame_ranges[view_idx][-1]
        frame_length = last_frame_end - last_frame_start
        query_regions.append(
            _QueryRegion(q_offset, q_offset + frame_length, view_idx, False, last_frame_start)
        )
        q_offset += frame_length

    traj_start_kv, traj_end_kv = traj_text_range
    traj_length = traj_end_kv - traj_start_kv
    query_regions.append(
        _QueryRegion(q_offset, q_offset + traj_length, num_views, True, traj_start_kv)
    )

    query_length = q_offset + traj_length
    min_val = torch.finfo(dtype).min
    attention_mask = torch.full(
        (1, 1, query_length, kv_length), min_val, dtype=dtype, device=device
    )

    for region in query_regions:
        rows = attention_mask[:, :, region.q_start : region.q_end, :]
        rows[:, :, :, :system_end] = 0

        for view_start, view_end in view_ranges[: region.num_attended_views]:
            rows[:, :, :, view_start:view_end] = 0
        if not region.is_traj_text:
            view_idx = region.num_attended_views  # a view region's own index
            if extended_vision_ranges:
                own_start = view_ranges[view_idx][0]
                rows[:, :, :, own_start : region.kv_start] = 0
            else:
                for frame_start, frame_end in frame_ranges[view_idx][:-1]:
                    rows[:, :, :, frame_start:frame_end] = 0

        # Causal within the query region's own F3 (or Traj+Text) block.
        region_length = region.q_end - region.q_start
        q_indices = torch.arange(region_length, device=device).unsqueeze(1)
        kv_indices = torch.arange(region_length, device=device).unsqueeze(0)
        rows[:, :, :, region.kv_start : region.kv_start + region_length] = torch.where(
            kv_indices <= q_indices, 0.0, min_val
        )

    if valid_length < kv_length:
        attention_mask[:, :, :, valid_length:] = min_val

    return attention_mask


def convert_to_streaming_window(
    window: dict,
    vision_start_token_id: int,
    vision_end_token_id: int,
    num_views: int = 4,
    num_frames_per_view: int = 4,
) -> dict:
    """Convert an N-frame prefill window to the streaming form used by windows 1+.

    Keeps each view's last frame's ``[VS]...[VE]`` vision block plus everything after
    the final ``[VE]`` (traj tokens, user prompt), dropping earlier frames. Works for
    both Alpamayo 1 and 1.5 — v1.5 just has extra camera/frame-label tokens in the
    dropped gaps, which the kept-block selection naturally excludes.
    """
    tokenized = window["tokenized_data"]
    input_ids = tokenized["input_ids"][0]  # [seq_len]
    pixel_values = tokenized["pixel_values"]  # [total_patches, hidden]
    image_grid_thw = tokenized["image_grid_thw"]  # [num_frames, 3]

    vs_positions = (input_ids == vision_start_token_id).nonzero(as_tuple=True)[0].tolist()
    ve_positions = (input_ids == vision_end_token_id).nonzero(as_tuple=True)[0].tolist()
    num_frames = num_views * num_frames_per_view
    if len(vs_positions) != num_frames or len(ve_positions) != num_frames:
        raise ValueError(
            f"Expected {num_frames} vision blocks, got {len(vs_positions)} starts "
            f"and {len(ve_positions)} ends."
        )

    # Last frame index per view (e.g. 3, 7, 11, 15 for 4 views x 4 frames).
    keep_indices = [
        view_idx * num_frames_per_view + (num_frames_per_view - 1) for view_idx in range(num_views)
    ]

    keep_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    for frame_idx in keep_indices:
        keep_mask[vs_positions[frame_idx] : ve_positions[frame_idx] + 1] = True
    keep_mask[ve_positions[-1] + 1 :] = True
    new_input_ids = input_ids[keep_mask].unsqueeze(0)

    patch_cumsum = [0, *image_grid_thw.prod(dim=-1).cumsum(0).tolist()]
    kept_pixel_values = torch.cat(
        [pixel_values[patch_cumsum[i] : patch_cumsum[i + 1]] for i in keep_indices]
    )
    kept_grid = image_grid_thw[keep_indices]

    new_tokenized = {
        "input_ids": new_input_ids,
        "attention_mask": torch.ones_like(new_input_ids),
        "pixel_values": kept_pixel_values,
        "image_grid_thw": kept_grid,
    }

    return {**window, "tokenized_data": new_tokenized}


class StreamingMixin:
    """FlashDrive streaming-inference behaviour (vision frame caching)."""

    def setup_streaming(
        self,
        *,
        extended_vision_ranges: bool,
        num_views: int = 4,
        num_frames_per_view: int = 4,
    ) -> None:
        """Initialize the streaming-window state (filled by the first prefill).

        ``extended_vision_ranges`` selects the per-view KV ranges of the streaming
        attention mask; see :func:`create_streaming_attention_mask_sdpa`.
        """
        self._extended_vision_ranges = extended_vision_ranges
        self._num_views = num_views
        self._num_frames_per_view = num_frames_per_view
        self._frame_ranges = None
        self._traj_text_range = None
        self._shift_src_index = None
        self._shift_dst_index = None
        self.streaming_position_ids = None
        self.streaming_cache_position = None
        self._cached_streaming_attention_mask = None

    def _locate_prompt_ranges(
        self, input_ids: torch.Tensor
    ) -> tuple[list[list[tuple[int, int]]], tuple[int, int]]:
        """Locate the prompt's layout: per-view frame ranges + the traj/text range.

        Returns ``(frame_ranges, traj_text_range)``: per-view lists of ``[VS]..[VE]``
        frame spans (end-exclusive), and the (start, end) of everything after the
        last vision block.
        """
        frame_ranges = [[] for _ in range(self._num_views)]

        vision_starts = torch.where(input_ids == self.vision_start_token_id)[1]
        vision_ends = torch.where(input_ids == self.vision_end_token_id)[1]

        for frame_idx, (vision_start, vision_end) in enumerate(
            zip(vision_starts, vision_ends, strict=True)
        ):
            view_idx = frame_idx // self._num_frames_per_view
            frame_ranges[view_idx].append((vision_start.item(), vision_end.item() + 1))

        traj_text_range = (vision_ends[-1].item() + 1, self.prefill_seq_length)
        return frame_ranges, traj_text_range

    def _shift_streaming_kv_cache(self) -> None:
        """Shift each view's KV cache: move frame blocks 1..N-1 into slots 0..N-2.

        One gather + scatter per tensor per layer (index tensors built at first
        prefill); the shift runs every window, so launch count matters.
        """
        src, dst = self._shift_src_index, self._shift_dst_index
        for layer in self._past_key_values.layers:
            layer.keys.index_copy_(2, dst, layer.keys.index_select(2, src))
            layer.values.index_copy_(2, dst, layer.values.index_select(2, src))

    def _create_cache_position(self) -> torch.Tensor:
        """Cache positions of a streaming window: each view's last-frame slot + traj/text."""
        cache_position = [
            torch.arange(start, end)
            for start, end in (view_frames[-1] for view_frames in self._frame_ranges)
        ]
        cache_position.append(torch.arange(self._traj_text_range[0], self._traj_text_range[1]))
        return torch.cat(cache_position, dim=0)

    def _ensure_streaming_attention_mask(self) -> torch.Tensor:
        """Build (and cache) the streaming attention mask for the active backbone."""
        if self._cached_streaming_attention_mask is None:
            self._cached_streaming_attention_mask = create_streaming_attention_mask_sdpa(
                frame_ranges=self._frame_ranges,
                traj_text_range=self._traj_text_range,
                kv_length=self.max_cache_len,
                valid_length=self.prefill_seq_length,
                device=self.streaming_position_ids.device,
                dtype=torch.bfloat16,
                extended_vision_ranges=self._extended_vision_ranges,
            )
        return self._cached_streaming_attention_mask

    def _first_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        """Run the full first-window prefill: cache KV, the padded position table,
        and the per-view frame ranges that every later streaming window reuses."""
        device = input_ids.device
        image_embeds, deepstack_image_embeds = self.vlm.model.visual(
            pixel_values, grid_thw=image_grid_thw
        )
        position_ids, _ = self.vlm.model.get_rope_index(input_ids, image_grid_thw, None, None)

        inputs_embeds, image_mask = self._embed_tokens_with_images(input_ids, image_embeds)

        padding_length = self.max_cache_len - input_ids.shape[1]
        if padding_length > 0:
            last_pos = position_ids[:, :, -1:]
            padding_pos = last_pos + torch.arange(1, padding_length + 1, device=device)
            position_ids = torch.cat([position_ids, padding_pos], dim=-1)

        self.streaming_position_ids = position_ids
        self._frame_ranges, self._traj_text_range = self._locate_prompt_ranges(input_ids)

        # Constant for the rest of the stream: the shift index pair (dst slots
        # <- src slots for every view's frame blocks) and the streaming window's
        # cache positions.
        src_index, dst_index = [], []
        for view_frames in self._frame_ranges:
            for (dst_start, dst_end), (src_start, src_end) in zip(
                view_frames[:-1], view_frames[1:], strict=True
            ):
                dst_index.append(torch.arange(dst_start, dst_end))
                src_index.append(torch.arange(src_start, src_end))
        self._shift_src_index = torch.cat(src_index).to(device)
        self._shift_dst_index = torch.cat(dst_index).to(device)
        self.streaming_cache_position = self._create_cache_position().to(device)

        self.vlm.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
            cache_position=torch.arange(input_ids.shape[1], device=device),
            visual_pos_masks=image_mask[..., 0],
            deepstack_visual_embeds=deepstack_image_embeds,
            streaming_attention_mask=None,
            use_cache=True,
        )

        # The 16-frame first prefill leaves shape-specific caches behind; later
        # windows run 4 frames, so reset them.
        self.vlm.model.visual.reset_shape_caches()
        self.vlm.model.language_model.reset_shape_caches()
