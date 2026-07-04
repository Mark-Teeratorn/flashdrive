"""ParoQuant W4A8 (INT4 weight + INT8 activation) quantization for inference.

The rotation + quantized linear modules and ``ParoQuantMixin``, whose
``setup_paroquant`` swaps the model's linears for the Marlin W4A8 path and fills
them from a pre-quantized checkpoint. Built on the external ``paroquant``
package's rotation CUDA kernel (imported lazily at setup since it JIT-builds)
and vLLM's Marlin kernels.

Not built on paroquant's own transformers backend: that path is W4A16 on
AutoAWQ's GEMM with flat checkpoint keys, while FlashDrive needs W4A8 on vLLM
Marlin in the nested layout the ``-PARO`` checkpoints use.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Rotation(nn.Module):
    """Scaled pairwise rotation applied to activations before the quantized linear.

    The paroquant kernel supports exactly 8 rotation layers over 128-channel groups.
    Buffers start at deterministic identity defaults (zero angles, in-order pairs,
    unit scales) and are filled from the checkpoint.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 128 == 0

        theta = torch.zeros(8, dim // 2, dtype=torch.float16)
        pairs = torch.arange(128, dtype=torch.int).repeat(dim // 128)
        pairs = pairs.unsqueeze(0).expand(8, -1)
        channel_scales = torch.ones(1, dim, dtype=torch.float16)

        self.register_buffer("theta", theta.contiguous())
        self.register_buffer("pairs", pairs.short().contiguous())
        self.register_buffer("channel_scales", channel_scales.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.rotation.rotate(x, self.pairs, self.theta, self.channel_scales, 128)


class MarlinW4A8Linear(nn.Module):
    """INT4 weight + per-token INT8 activation linear via vLLM Marlin kernel.

    Lifecycle (inference):
        1. create instance (empty Marlin buffers)
        2. load pre-converted Marlin buffers from the checkpoint via
           ``load_state_dict`` (see ``_load_from_state_dict``)
        3. forward(x)  — dynamically quantises x to INT8, runs Marlin GEMM
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        from vllm.scalar_type import scalar_types

        self.quant_type = scalar_types.uint4

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("qzeros", torch.empty(0, dtype=torch.int32))
        self.register_buffer("workspace", torch.empty(0, dtype=torch.int32))
        self.register_buffer("g_idx", torch.empty(0, dtype=torch.int32))
        self.register_buffer("g_idx_sort_indices", torch.empty(0, dtype=torch.int32))
        self.register_buffer("input_global_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("bias", None)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Override to allow loading Marlin buffers whose shapes differ from
        the empty(0) placeholders created in __init__."""
        handled = {}
        for name in list(self._buffers.keys()):
            key = prefix + name
            if key in state_dict:
                self._buffers[name] = state_dict.pop(key)
                handled[name] = self._buffers.pop(name)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._buffers.update(handled)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lazy (like the vllm import in __init__): resolves from sys.modules per
        # call in eager mode, and is baked into the graph under torch.compile.
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_awq_marlin_linear,
        )

        return apply_awq_marlin_linear(
            input=x,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            quant_type=self.quant_type,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            input_global_scale=self.input_global_scale,
            bias=self.bias,
            input_dtype=torch.int8,
        )


class RotateLinearW4A8(nn.Module):
    """Rotation followed by the Marlin W4A8 linear; buffers filled from the checkpoint."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rotation = Rotation(in_features)
        self.qlinear = MarlinW4A8Linear(in_features, out_features)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rotation(x)
        x = self.qlinear(x)
        return x


class BF16RotateLinearWrapper(nn.Module):
    """Runs the fp16 rotation + Marlin path inside a bf16 model (autocast-exempt)."""

    def __init__(self, rotate_linear: nn.Module) -> None:
        super().__init__()
        self.rotate_linear = rotate_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            out = self.rotate_linear(x.to(torch.float16))
        return out.to(input_dtype)


def _iter_linears(module: nn.Module) -> Iterator[tuple[nn.Module, str, nn.Linear]]:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            yield module, name, child
        yield from _iter_linears(child)


class ParoQuantMixin:
    """FlashDrive W4A8 quantization behaviour (checkpoint-backed module surgery)."""

    def setup_paroquant(self, checkpoint_path: str) -> None:
        """Quantize the VLM language model from a pre-quantized PARO checkpoint.

        Swaps every language-model linear for the rotation + Marlin W4A8 path, then
        fills the buffers from ``checkpoint_path`` (a local dir or hub repo). The
        action expert stays in unquantized bf16 by design.
        """
        # Deferred: importing paroquant JIT-builds its CUDA kernel (needs nvcc);
        # it registers torch.ops.rotation.rotate, which Rotation.forward uses.
        import paroquant.kernels.cuda  # noqa: F401
        from safetensors.torch import load_file

        for layer in self.vlm.model.language_model.layers:
            for parent, name, linear in list(_iter_linears(layer)):
                quantized = RotateLinearW4A8(linear.in_features, linear.out_features)
                setattr(parent, name, BF16RotateLinearWrapper(quantized))

        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.is_dir():
            from huggingface_hub import snapshot_download

            checkpoint_dir = Path(snapshot_download(checkpoint_path))
        safetensor_files = sorted(checkpoint_dir.glob("model*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No model*.safetensors shards found in {checkpoint_dir}.")
        state_dict = {}
        for shard in safetensor_files:
            state_dict.update(load_file(str(shard), device="cpu"))

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            quant_missing = [k for k in missing if "qlinear" in k or "rotation" in k]
            if quant_missing:
                logger.warning(
                    f"Missing quantization keys ({len(quant_missing)}): {quant_missing[:5]}"
                )
            else:
                logger.info(f"Missing keys ({len(missing)}, all non-quant — OK)")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        del state_dict

        # The surgery builds its modules on CPU and the checkpoint loads CPU tensors;
        # bring the quantized language model to the model's device.
        self.vlm.model.language_model.to(next(self.parameters()).device)
