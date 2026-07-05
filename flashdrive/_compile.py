"""torch.compile + cudagraph machinery for FlashDrive.

`TorchCompileMixin` owns the buffer-backed compiled-primitive driver behind every
hot-path forward (``_encode`` / ``_action``, the DFlash prefill / block-step /
traj-forward steps). Compilation is lazy — one cudagraph per primitive — under
the mode fixed at load time by ``setup_torch_compile``.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _CompiledStep:
    """One buffer-backed primitive: persistent input buffers + the step bound to them.

    ``fn`` is argument-free (the primitive bound to its buffers); ``compiled``
    starts as None and is filled on first dispatch under a compile mode (after
    one eager warmup call).
    """

    buffers: dict[str, Any]
    fn: Callable[[], Any]
    compiled: Callable[[], Any] | None = None


class TorchCompileMixin:
    """Buffer-backed ``torch.compile`` + cudagraph dispatch for hot-path primitives."""

    def setup_torch_compile(self, mode: str | None) -> None:
        """Fix the ``torch.compile`` mode for the rollout's compiled primitives.

        ``mode`` is a ``torch.compile`` mode string (e.g. ``"max-autotune"``) or
        ``None`` for eager. Also initializes the primitive registry, so it must
        run before any primitive (and rerunning it resets them all).
        """
        self._torch_compile = mode
        self._compiled_step_registry: dict[str, _CompiledStep] = {}

    def has_compiled_step(self, key: str) -> bool:
        """True once a primitive has run (and therefore compiled, if enabled)."""
        return key in self._compiled_step_registry

    def _compiled_step(
        self,
        key: str,
        tensors: dict[str, Any],
        fn: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """Run one buffer-backed compiled primitive: ``fn(buffers)``.

        The first call per ``key`` allocates persistent input buffers shaped like
        ``tensors`` and binds ``fn`` to them; later calls reuse that binding (their
        ``fn`` is ignored, so whatever ``fn`` closes over is fixed by the first
        call). Every call copies the current inputs into the buffers and runs the
        bound step — eager when ``_torch_compile`` is None, otherwise lazily
        compiled into one cudagraph per key.

        ``tensors`` values may be a Tensor, a list/tuple of Tensors, or ``None``
        (an absent optional input); ``buffers`` mirrors that structure.
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
            step = _CompiledStep(buffers=buffers, fn=functools.partial(fn, buffers))
            self._compiled_step_registry[key] = step

        for name, t in tensors.items():
            if t is None:
                continue
            buf = step.buffers[name]
            if isinstance(buf, list):
                for b, e in zip(buf, t, strict=True):
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
