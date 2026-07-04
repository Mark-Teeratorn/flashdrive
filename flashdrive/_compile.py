"""torch.compile + cudagraph machinery for FlashDrive.

`TorchCompileMixin` is infrastructure, not a feature: its setup takes a mode,
not a checkpoint. It owns the buffer-backed compiled-primitive driver that every
hot-path forward runs through (``_encode`` / ``_action`` on the base, the DFlash
prefill / block-step / text-forward steps). Compilation is lazy — one cudagraph per
primitive and input shape — under the mode fixed at load time by
:meth:`TorchCompileMixin.setup_torch_compile`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _CompiledStep:
    """One buffer-backed primitive: persistent input buffers + its closure.

    ``compiled`` starts as None and is filled on first dispatch under a compile
    mode (after one eager warmup call).
    """

    buffers: dict[str, Any]
    fn: Callable[[], Any]
    compiled: Callable[[], Any] | None = None


class TorchCompileMixin:
    """Buffer-backed ``torch.compile`` + cudagraph dispatch for hot-path primitives."""

    def setup_torch_compile(self, mode: str | None) -> None:
        """Fix the ``torch.compile`` mode for the rollout's compiled primitives.

        ``mode`` is a ``torch.compile`` mode string (e.g. ``"max-autotune"``) or
        ``None`` to run everything eagerly. This is load-time configuration, not
        a per-call knob: primitives compile lazily on first use and cache their
        compiled form. Initializes the mixin's whole state — rerunning it resets
        the primitive registry, and no primitive may run before it.
        """
        self._torch_compile = mode
        self._compiled_step_registry: dict[str, _CompiledStep] = {}

    def _compiled_step(
        self,
        key: str,
        tensors: dict[str, Any],
        make_fn: Callable[[dict[str, Any]], Callable[[], Any]],
    ) -> Any:
        """Run one buffer-backed compiled primitive.

        On first call (per ``key``) it allocates persistent input buffers shaped
        like ``tensors``, builds the closure once via ``make_fn(buffers)``, and
        registers both. Every call then copies the current inputs into those
        buffers and runs the closure — eagerly when compilation is disabled
        (``_torch_compile is None``), otherwise via its lazily compiled form
        inside a fresh cudagraph step. Keying by ``key`` gives one cudagraph per
        input shape.

        ``tensors`` values may be a Tensor, a list/tuple of Tensors (e.g. deepstack
        embeds), or ``None`` (an absent optional input); ``buffers`` mirrors that
        structure so the closure can index it by name.
        """
        step = self._compiled_step_registry.get(key)
        if step is None:
            buffers: dict[str, Any] = {}
            for name, t in tensors.items():
                if t is None:
                    buffers[name] = None
                elif isinstance(t, (list, tuple)):
                    buffers[name] = [torch.empty_like(e) for e in t]
                else:
                    buffers[name] = torch.empty_like(t)
            step = _CompiledStep(buffers=buffers, fn=make_fn(buffers))
            self._compiled_step_registry[key] = step

        for name, t in tensors.items():
            if t is None:
                continue
            buf = step.buffers[name]
            if isinstance(buf, list):
                for b, e in zip(buf, t):
                    b.copy_(e)
            else:
                buf.copy_(t)

        if self._torch_compile is None:
            return step.fn()
        if step.compiled is None:
            step.fn()  # Warmup
            step.compiled = torch.compile(step.fn, mode=self._torch_compile, fullgraph=True)
        torch.compiler.cudagraph_mark_step_begin()
        return step.compiled()
