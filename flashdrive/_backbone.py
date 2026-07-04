"""FlashDrive replacements for the Qwen3-VL backbone runtime.

Drop-in patched vision/text modules applied by ``patch_backbone`` (streaming
RoPE + attention mask, fuse-aware projections, DFlash hidden-state capture,
fullgraph-compilable), and the ``StaticCache`` shared by every FlashDrive
rollout. A torch 2.9 Conv3D -> Linear patch-embed fix is applied globally at
import time (see the bottom of the module).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.cache_utils as cache_utils
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl
from transformers import PretrainedConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb_vision,
    rotate_half,
)


class StaticLayer(cache_utils.StaticLayer):
    """Upstream ``StaticLayer`` with a pre-sized batch and partial-batch updates.

    Everything cudagraph-related (allocation, ``mark_static_address``, mask sizes,
    sequence-length accounting) is inherited. FlashDrive adds what its rollout
    needs: the cache is allocated for the full multi-sample decode batch even when
    the batch-1 prefill initializes it, and ``expand_batch`` fans the prompt KV out
    to every sample.
    """

    def __init__(self, max_cache_len: int, max_batch_size: int = 1) -> None:
        super().__init__(max_cache_len=max_cache_len)
        self._max_batch_size = max_batch_size

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        # Allocate for the configured decode batch even when the batch-1 prefill
        # triggers initialization (expand is a zero-copy view; upstream only reads
        # its shape/dtype/device).
        batch = max(self._max_batch_size, key_states.shape[0])
        super().lazy_initialization(key_states.expand(batch, -1, -1, -1))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the caches in place and return the incoming batch's slots.

        The source batch may be smaller than the cache's (the batch=1 prefill
        writing into a multi-sample cache); writes and reads always cover the
        leading ``src_batch`` slots, which is the full cache during decode.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        cache_position = cache_kwargs["cache_position"]
        batch = key_states.shape[0]
        self.keys[:batch].index_copy_(2, cache_position, key_states)
        self.values[:batch].index_copy_(2, cache_position, value_states)
        return self.keys[:batch], self.values[:batch]

    def expand_batch(self) -> None:
        """Copy batch 0's KV to all other batch slots.

        Call this after prefill (batch=1) and before multi-sample decode,
        so all samples start with the same prompt KV.
        """
        if self.is_initialized and self.max_batch_size > 1:
            self.keys[1:].copy_(self.keys[:1].expand(self.max_batch_size - 1, -1, -1, -1))
            self.values[1:].copy_(self.values[:1].expand(self.max_batch_size - 1, -1, -1, -1))


class StaticCache(cache_utils.Cache):
    """Pre-allocated fixed-size KV cache: one :class:`StaticLayer` per decoder layer."""

    def __init__(
        self, config: PretrainedConfig, max_cache_len: int, max_batch_size: int = 1
    ) -> None:
        config = config.get_text_config(decoder=True)
        super().__init__(
            layers=[
                StaticLayer(max_cache_len=max_cache_len, max_batch_size=max_batch_size)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def expand_batch(self) -> None:
        """Copy batch 0's KV to all batch slots. Call before multi-sample decode."""
        for layer in self.layers:
            layer.expand_batch()


_PATCHED_CLASSES: dict[type, type] = {}


def _drop_in(cls: type) -> type:
    """Register a patched class as the drop-in replacement for its direct base.

    Keyed by the exact class (both Alpamayo backbones instantiate transformers'
    own Qwen3-VL modules): no name-based false matches, and idempotent since a
    patched module's type is never a key.
    """
    _PATCHED_CLASSES[cls.__base__] = cls
    return cls


class Qwen3VLVisionPatchEmbed(qwen3vl.Qwen3VLVisionPatchEmbed):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        nn.Module.__init__(self)
        self.in_features = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.proj = nn.Linear(self.in_features, config.hidden_size, bias=True)

        # Hook to convert Conv3d weights [out, in, t, h, w] -> Linear [out, in*t*h*w]
        def convert_conv3d_weights(
            state_dict: dict[str, torch.Tensor], prefix: str, *args: Any
        ) -> None:
            weight_key = prefix + "weight"
            if weight_key in state_dict and state_dict[weight_key].ndim == 5:
                state_dict[weight_key] = state_dict[weight_key].flatten(1).contiguous()

        self.proj._register_load_state_dict_pre_hook(convert_conv3d_weights, with_module=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.reshape(-1, self.in_features))


@_drop_in
class Qwen3VLVisionAttention(qwen3vl.Qwen3VLVisionAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]

        if not hasattr(self, "_num_chunks"):
            self._num_chunks = cu_seqlens.numel() - 1

        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, -1)
        query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)

        query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)

        chunk_size = seq_len // self._num_chunks
        query = query.reshape(self._num_chunks, chunk_size, self.num_heads, -1).transpose(1, 2)
        key = key.reshape(self._num_chunks, chunk_size, self.num_heads, -1).transpose(1, 2)
        value = value.reshape(self._num_chunks, chunk_size, self.num_heads, -1).transpose(1, 2)

        output = F.scaled_dot_product_attention(query, key, value, scale=self.scaling)
        output = self.proj(output.transpose(1, 2).reshape(seq_len, -1))

        return output


@_drop_in
class Qwen3VLVisionModel(qwen3vl.Qwen3VLVisionModel):
    def reset_shape_caches(self) -> None:
        """Drop the shape-specific forward caches (call when the frame count changes)."""
        for attr in ("_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens"):
            if hasattr(self, attr):
                delattr(self, attr)
        for block in self.blocks:
            if hasattr(block.attn, "_num_chunks"):
                del block.attn._num_chunks

    def _init_shape_caches(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> None:
        seq_len = hidden_states.size(0)

        self._cached_pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        rotary_emb = self.rot_pos_emb(grid_thw).reshape(seq_len, -1)
        rotary_emb = torch.cat((rotary_emb, rotary_emb), dim=-1)
        self._cached_position_embeddings = (rotary_emb.cos(), rotary_emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
        self._cached_cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)

        if not hasattr(self, "_cached_pos_embeds"):
            self._init_shape_caches(hidden_states, grid_thw)

        hidden_states = (hidden_states + self._cached_pos_embeds).reshape(hidden_states.size(0), -1)

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=self._cached_cu_seqlens,
                position_embeddings=self._cached_position_embeddings,
            )
            if layer_idx in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(layer_idx)
                deepstack_features.append(self.deepstack_merger_list[merger_idx](hidden_states))
        return self.merger(hidden_states), deepstack_features


def apply_rotary_pos_emb_single(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embeddings to a single tensor (query or key)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (tensor * cos) + (rotate_half(tensor) * sin)


@_drop_in
class Qwen3VLTextAttention(qwen3vl.Qwen3VLTextAttention):
    """Patched Qwen3VL text attention: streaming RoPE + optional fused QKV projection.

    Keys/values are cached un-roped; RoPE is applied per-query against the cached
    (padded) position table, which is what lets the streaming KV shift work.
    """

    # Class default; setup_expert_fusion sets the instance attribute to True.
    fuse_qkv = False

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: cache_utils.Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.fuse_qkv:
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.reshape(hidden_shape)).transpose(1, 2)
        value_states = value_states.reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        cos_q, sin_q = cos[:, cache_position, :], sin[:, cache_position, :]

        # Store the un-roped keys and values in cache
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        query_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@_drop_in
class Qwen3VLTextMLP(qwen3vl.Qwen3VLTextMLP):
    """Qwen3VL text MLP with optional fused gate/up projection."""

    # Class default; setup_expert_fusion sets the instance attribute to True.
    fuse_gate_up = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fuse_gate_up:
            gate, up = self.gate_up_proj(x)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)


@_drop_in
class Qwen3VLTextModel(qwen3vl.Qwen3VLTextModel):
    def reset_shape_caches(self) -> None:
        """Drop the shape-specific forward caches (call when the frame count changes)."""
        if hasattr(self, "_cached_deepstack_indices"):
            del self._cached_deepstack_indices

    def set_capture_layer_ids(self, layer_ids: list[int]) -> None:
        """Configure which layer hidden states to return in ``hidden_states``.

        torch.compile-friendly: the set is a constant determined before
        compilation, so ``layer_idx in set`` traces as static control flow.
        """
        self._capture_layer_ids = set(layer_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: cache_utils.Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        streaming_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        position_ids = position_ids[0]

        if inputs_embeds.shape[1] > 1:  # prefill, attention handles decode by default
            if streaming_attention_mask is not None:
                attention_mask = streaming_attention_mask
            else:
                attention_mask = create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )

        if deepstack_visual_embeds is not None and not hasattr(self, "_cached_deepstack_indices"):
            self._cached_deepstack_indices = visual_pos_masks.flatten().nonzero(as_tuple=True)[0]

        hidden_states = inputs_embeds
        captured: list[torch.Tensor] = []
        capture_ids = getattr(self, "_capture_layer_ids", ())
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # Capture specific layers for DFlash draft context (compile-friendly)
            if layer_idx in capture_ids:
                captured.append(hidden_states)
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
                flat_hidden.index_add_(
                    0, self._cached_deepstack_indices, deepstack_visual_embeds[layer_idx]
                )

        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values,
            hidden_states=tuple(captured) if captured else None,
        )


def patch_backbone(model: nn.Module) -> None:
    """Swap the Qwen3-VL modules for their FlashDrive replacements.

    The patched forwards carry the streaming attention-mask / RoPE path, the
    fuse-aware projections, and the DFlash hidden-state capture, and are
    fullgraph-compilable. Every ``@_drop_in`` class is a drop-in subclass (same
    parameters and buffers), so patching is a pure in-place ``__class__``
    reassignment: no weight copies, no device moves.
    """
    for module in model.modules():
        patched = _PATCHED_CLASSES.get(type(module))
        if patched is not None:
            module.__class__ = patched


# Applied at import time (the stock baseline path needs it too): the Conv3d patch_embed
# is extremely slow on Blackwell with torch 2.9, so swap in the Linear version globally.
# Still needed as of transformers 4.57 (upstream remains Conv3d-based).
qwen3vl.Qwen3VLVisionPatchEmbed = Qwen3VLVisionPatchEmbed
