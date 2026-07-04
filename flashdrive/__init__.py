"""Compose the FlashDrive feature patches onto a base Alpamayo model.

FlashDrive never edits upstream code; it patches at runtime, one mechanism per
level:

* the model instance: :func:`from_pretrained` synthesizes a subclass
  mixing the feature mixins ahead of the original class in the MRO;
* backbone submodules: ``patch_backbone`` swaps each Qwen3-VL module's
  ``__class__`` to a registered drop-in subclass (:mod:`flashdrive._backbone`);
* the Conv3D patch-embed: replaced module-globally at import time, the one
  patch the stock baseline path needs too (:mod:`flashdrive._backbone`).

Each mixin lives in its own module:

* :class:`flashdrive._base.FlashDriveBaseMixin` — shared scaffold
* :class:`flashdrive._compile.TorchCompileMixin` — torch.compile/cudagraph machinery
* :class:`flashdrive.streaming.StreamingMixin` — streaming
* :class:`flashdrive.dflash.DFlashMixin` — speculative decoding
* :class:`flashdrive.quantization.ParoQuantMixin` — W4A8 quantization
* :class:`flashdrive.fusion.ExpertFusionMixin` — expert projection fusion
"""

from typing import NamedTuple

import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import PretrainedConfig

from flashdrive._base import FlashDriveBaseMixin
from flashdrive._compile import TorchCompileMixin
from flashdrive.dflash import DFlashMixin
from flashdrive.fusion import ExpertFusionMixin
from flashdrive.quantization import ParoQuantMixin
from flashdrive.streaming import StreamingMixin


class FlashDriveMixin(
    FlashDriveBaseMixin,
    TorchCompileMixin,
    StreamingMixin,
    DFlashMixin,
    ParoQuantMixin,
    ExpertFusionMixin,
):
    """All FlashDrive feature patches composed into a single mixin.

    Method names are disjoint across the mixins, so composition order is
    irrelevant. Each mixin contributes one ``setup_*`` method;
    :func:`from_pretrained` applies them in dependency order
    (``setup_rollout`` first — it patches the backbone forwards the other
    setups build on).

    Isolation: mixins never call each other or touch each other's private
    state — the driver mediates. Shared state has one writer: the base owns
    ``_past_key_values``, ``prefill_seq_length``, ``max_cache_len``, and the
    prompt token ids; streaming publishes ``streaming_position_ids`` and
    ``streaming_cache_position``; dflash publishes ``dflash_block_size``.
    """


class _BackboneTraits(NamedTuple):
    """Where the supported Alpamayo backbones diverge.

    extended_vision_ranges: the Cosmos (Alpamayo 1.5) prompt places camera-name /
        frame-label text between vision blocks, so the streaming attention mask must
        extend each view's KV range back over those tokens. Qwen3-VL (Alpamayo 1 / R1)
        packs frames back-to-back and uses tight per-frame ranges.
    force_sdpa: DFlash's multi-token block verify builds its mask via
        ``create_causal_mask``, whose result under flash_attention_2 vs sdpa is
        consumed differently by the two backbones. Empirically Cosmos needs the
        explicit sdpa mask (flash_attention_2 -> None -> non-causal block -> garbage),
        while Qwen3-VL needs flash_attention_2 (sdpa's explicit mask makes it emit an
        empty CoT).
    """

    extended_vision_ranges: bool
    force_sdpa: bool


# One row per supported backbone; everything downstream branches on these traits,
# never on a version identity. Supporting a new Alpamayo release is one new row.
_BACKBONE_TRAITS = {
    AlpamayoR1: _BackboneTraits(extended_vision_ranges=False, force_sdpa=False),
    Alpamayo1_5: _BackboneTraits(extended_vision_ranges=True, force_sdpa=True),
}


def resolve_model_class(model_path: str) -> type:
    """Resolve the Alpamayo model class from a checkpoint's ``model_type``."""
    model_type = PretrainedConfig.get_config_dict(model_path)[0]["model_type"]
    model_classes = {cls.config_class.model_type: cls for cls in _BACKBONE_TRAITS}
    if model_type not in model_classes:
        raise ValueError(
            f"Unsupported model type {model_type!r}; supported: {sorted(model_classes)}."
        )
    return model_classes[model_type]


def from_pretrained(
    base_model_path: str,
    device: str = "cuda",
    torch_compile: str | None = "max-autotune",
) -> torch.nn.Module:
    """Load the fully-optimized FlashDrive model from its base checkpoint.

    The model class is resolved from the checkpoint's ``model_type``; the quantized
    and draft checkpoints are derived from the base by suffix (``-PARO`` /
    ``-DFlash``). Mixes the FlashDrive rollout onto the model, patches the backbone
    modules, quantizes the language model (W4A8), fuses the action expert's
    projections, and attaches the DFlash draft model. ``torch_compile`` fixes the
    compile mode for the rollout's primitives (``None`` = eager, e.g. for probing).
    """
    model_cls = resolve_model_class(base_model_path)
    traits = _BACKBONE_TRAITS[model_cls]

    model = model_cls.from_pretrained(base_model_path, dtype=torch.bfloat16)
    model = model.to(device).eval()

    model.__class__ = type(f"FlashDrive{model_cls.__name__}", (FlashDriveMixin, model_cls), {})
    model.setup_rollout(force_sdpa=traits.force_sdpa)
    model.setup_torch_compile(torch_compile)
    model.setup_streaming(extended_vision_ranges=traits.extended_vision_ranges)
    model.setup_paroquant(f"{base_model_path}-PARO")
    model.setup_expert_fusion()
    model.setup_dflash(f"{base_model_path}-DFlash")
    return model


__all__ = ["from_pretrained", "resolve_model_class"]
