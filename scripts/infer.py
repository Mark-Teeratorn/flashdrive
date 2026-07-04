"""End-to-end inference: load a clip, run the model, report minADE + per-step latency.

The checkpoint org selects the mode: z-lab checkpoints run the optimized FlashDrive
stack (streaming + DFlash + W4A8 + expert fusion + action caching + compile), while
upstream (nvidia) checkpoints run the stock model, for before/after timing. The
Alpamayo release (1 vs 1.5) is resolved from the checkpoint's config.
"""

import argparse
import functools
import importlib
import inspect
import logging
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import physical_ai_av
import torch

# Importing flashdrive also applies, as a side effect, the torch 2.9 Conv3D->Linear
# fix (in flashdrive._backbone) that the stock baseline path relies on too.
import flashdrive
from flashdrive.streaming import convert_to_streaming_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Enable TF32 matmul for better perf on Ampere+ GPUs.
torch.set_float32_matmul_precision("high")

# Fixed optimized-path diffusion config.
MAX_NEW_TOKENS = 128
DIFFUSION_STEPS = 8
CACHE_STEPS = [3, 4, 5, 6]


def _is_optimized(model_path: str) -> bool:
    """True for a z-lab (optimized) checkpoint; False for a stock (nvidia) baseline."""
    return model_path.split("/", 1)[0].lower() == "z-lab"


def load_helper(model_path: str) -> ModuleType:
    """Import the ``helper`` module of the checkpoint's Alpamayo package.

    The package is derived from the model class that :func:`resolve_model_class`
    reads out of the checkpoint's config -- no name conventions involved.
    """
    package = flashdrive.resolve_model_class(model_path).__module__.split(".")[0]
    return importlib.import_module(f"{package}.helper")


def create_inputs(helper: ModuleType, args: argparse.Namespace, processor: Any) -> list[dict]:
    """Build per-window model inputs (up front, so data loading stays out of the timed loop).

    Optimized (streaming) runs keep window 0 as the full 16-frame prefill and convert
    windows 1+ to the 4-frame streaming form; the stock baseline uses full windows.
    """
    package = helper.__name__.rsplit(".", 1)[0]
    load_dataset = importlib.import_module(
        f"{package}.load_physical_aiavdataset"
    ).load_physical_aiavdataset
    # Alpamayo 1.5 conditions its prompt on camera indices; detect from the helper.
    camera_conditioned = "camera_indices" in inspect.signature(helper.create_message).parameters

    streaming = _is_optimized(args.model_path)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ego_keys = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
    inputs_list = []
    for window_idx in range(args.num_steps):
        t0 = args.t0_us + window_idx * args.time_step_us
        data = load_dataset(args.clip_id, t0_us=t0, num_frames=4, avdi=avdi)
        frames = data["image_frames"].flatten(0, 1)  # (4, 4, C, H, W) -> (16, C, H, W)

        if camera_conditioned:
            messages = helper.create_message(frames, camera_indices=data["camera_indices"])
        else:
            messages = helper.create_message(frames)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {"tokenized_data": inputs, **{k: data[k] for k in ego_keys}}
        if streaming and window_idx > 0:
            model_inputs = convert_to_streaming_window(
                model_inputs, vision_start_token_id, vision_end_token_id
            )
        inputs_list.append(model_inputs)

    return inputs_list


def compute_min_ade(gt_future_xyz: torch.Tensor, pred_xyz: torch.Tensor) -> tuple[float, int]:
    """Return (min ADE, sample index) of the predicted trajectories vs ground truth."""
    gt_xy = gt_future_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return float(diff.min()), int(diff.argmin())


def make_sample_fn(
    helper: ModuleType, model: torch.nn.Module, args: argparse.Namespace
) -> Callable[[dict], tuple]:
    """Bind the per-run rollout once: mode, sampling kwargs, and device shipping.

    The returned callable maps ``model_inputs -> (pred_xyz, pred_rot, extra)``, so
    the benchmark loop stays mode-blind.
    """
    if _is_optimized(args.model_path):
        rollout = functools.partial(
            model.sample_trajectories_streaming,
            max_new_tokens=MAX_NEW_TOKENS,
            diffusion_kwargs={
                "inference_step": DIFFUSION_STEPS,
                "cache_steps": CACHE_STEPS,
                "int_method": "euler_with_cache",
            },
        )
    else:
        rollout = model.sample_trajectories_from_data_with_vlm_rollout

    def sample(model_inputs: dict) -> tuple:
        data = helper.to_device(model_inputs, args.device)
        return rollout(data=data, num_traj_samples=args.num_traj_samples, return_extra=True)

    return sample


@torch.inference_mode()
def run_inference(sample: Callable, model_inputs: dict, log: bool = True) -> float:
    """Run one rollout; return its minADE."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = sample(model_inputs)
    if pred_xyz is None:  # streaming first prefill only caches KV
        return float("inf")

    min_ade, min_ade_idx = compute_min_ade(model_inputs["ego_future_xyz"], pred_xyz)
    if log:
        logger.info(f"Chain-of-Causation:\n{extra['cot'][0][0][min_ade_idx]}")
        logger.info(f"MinADE: {min_ade}")
    return min_ade


def run_benchmark(sample: Callable, all_inputs: list[dict], warmup_steps: int) -> None:
    """Warm up, then time per-window inference and report minADE + latency stats."""
    logger.info("Warmup")
    for model_inputs in all_inputs[:warmup_steps]:
        run_inference(sample, model_inputs, log=False)
    logger.info("Warmup completed")

    min_ades, latencies = [], []
    for step, model_inputs in enumerate(all_inputs[warmup_steps:], warmup_steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        min_ades.append(run_inference(sample, model_inputs))
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        logger.info(f"Step {step}/{len(all_inputs)} completed | Latency: {latencies[-1]:.1f} ms")

    logger.info(f"Average MinADE: {np.mean(min_ades)}")
    logger.info(
        f"LATENCY_SUMMARY ms: mean={np.mean(latencies):.1f} median={np.median(latencies):.1f} "
        f"min={np.min(latencies):.1f} p90={np.percentile(latencies, 90):.1f} n={len(latencies)}"
    )


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Construct the model: the full optimized FlashDrive stack or a stock baseline."""
    if _is_optimized(args.model_path):
        # W4A8 (PARO) + FlashDrive patches + expert fusion + DFlash drafting, in one
        # call; the -PARO/-DFlash checkpoints are derived from the base by suffix.
        return flashdrive.from_pretrained(args.model_path, device=args.device)

    model_cls = flashdrive.resolve_model_class(args.model_path)
    model = model_cls.from_pretrained(args.model_path, dtype=torch.bfloat16)
    return model.to(args.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="z-lab/Alpamayo-1.5-10B",
        help="Base checkpoint. A z-lab checkpoint runs optimized FlashDrive; an upstream "
        "(nvidia) checkpoint runs the stock model. The Alpamayo release is resolved from "
        "the checkpoint's config; -PARO/-DFlash are derived from the base by suffix.",
    )
    parser.add_argument("--num-steps", type=int, default=120, help="Total windows to run.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Initial windows excluded from the metrics. Must be >= 1 on optimized runs: "
        "window 0 only prefills the streaming cache and returns no trajectories.",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1,
        help="Number of trajectory samples to draw per step.",
    )
    parser.add_argument(
        "--clip-id",
        default="87147a1b-3eef-4c25-94d2-ec7718a49a7a",
        help="PhysicalAI-AV clip to evaluate.",
    )
    parser.add_argument("--t0-us", type=int, default=1_700_000, help="Clip start time (us).")
    parser.add_argument("--time-step-us", type=int, default=100_000, help="Window stride (us).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    helper = load_helper(args.model_path)
    model = build_model(args)
    processor = helper.get_processor(model.tokenizer)
    sample = make_sample_fn(helper, model, args)

    logger.info("Loading inputs...")
    all_inputs = create_inputs(helper, args, processor)
    run_benchmark(sample, all_inputs, args.warmup_steps)


if __name__ == "__main__":
    main()
