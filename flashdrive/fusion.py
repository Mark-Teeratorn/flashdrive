"""Expert projection fusion for FlashDrive.

`ExpertFusionMixin` fuses the action expert's q/k/v and gate/up projections into
single GEMMs via `FusedLinear`. The fuse-aware forwards live on the patched
backbone classes (:mod:`flashdrive._backbone`); their fuse flags default to
False, and ``setup_expert_fusion`` flips them per instance after the surgery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLinear(nn.Module):
    """One matmul for several same-input linear projections (e.g. q/k/v or gate/up).

    Built directly from the projections it replaces: their weights are concatenated
    row-wise so ``[y1; y2; ...] = x @ [W1; W2; ...]^T``, and the forward splits the
    output back per projection.
    """

    def __init__(self, *linears: nn.Linear) -> None:
        super().__init__()
        self.output_sizes = [linear.out_features for linear in linears]
        with torch.no_grad():
            self.weight = nn.Parameter(torch.cat([linear.weight for linear in linears], dim=0))
            if linears[0].bias is not None:
                self.bias = nn.Parameter(torch.cat([linear.bias for linear in linears], dim=0))
            else:
                self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.split(F.linear(x, self.weight, self.bias), self.output_sizes, dim=-1)


class ExpertFusionMixin:
    """FlashDrive expert-fusion behaviour (fused q/k/v and gate/up projections)."""

    def setup_expert_fusion(self) -> None:
        """Fuse the action expert's q/k/v and gate/up projections for fewer GEMM launches.

        In-place and idempotent: each ``FusedLinear`` replaces the projections it
        was built from, and the module's fuse flag is flipped. Requires the
        backbone patched by ``setup_rollout`` (the fuse-aware forwards).
        """
        for layer in self.expert.layers:
            attn = layer.self_attn
            if not attn.fuse_qkv:
                attn.qkv_proj = FusedLinear(attn.q_proj, attn.k_proj, attn.v_proj)
                del attn.q_proj, attn.k_proj, attn.v_proj
                attn.fuse_qkv = True

            mlp = layer.mlp
            if not mlp.fuse_gate_up:
                mlp.gate_up_proj = FusedLinear(mlp.gate_proj, mlp.up_proj)
                del mlp.gate_proj, mlp.up_proj
                mlp.fuse_gate_up = True
