#!/usr/bin/env python3
"""FlashDrive Model Server for Alpamayo 1.5.

Listens over Unix Domain Socket or TCP for camera images and ego vehicle state from
the Bench2Drive / Fail2Drive agent, executes streaming inference with FlashDrive
(or dummy mode for pipeline validation), and returns predicted trajectory waypoints.
"""

import argparse
import functools
import importlib
import inspect
import logging
import os
import pickle
import socket
import struct
import sys
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Auto-configure CUDA_HOME and CPATH for torch JIT extensions to find nvidia CUDA headers (e.g. cusolverDn.h)
if "CUDA_HOME" not in os.environ and os.path.exists("/usr/local/cuda"):
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
if os.path.exists("/usr/local/cuda/bin") and "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"

import glob
nvidia_includes = glob.glob(os.path.join(os.path.dirname(sys.executable), "..", "lib", "python*", "site-packages", "nvidia", "*", "include"))
if nvidia_includes:
    cpath = ":".join(nvidia_includes) + ((":" + os.environ["CPATH"]) if "CPATH" in os.environ else "")
    os.environ["CPATH"] = cpath
    os.environ["CPLUS_INCLUDE_PATH"] = cpath

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FlashDriveServer] %(message)s",
)
logger = logging.getLogger("FlashDriveServer")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DEFAULT_SOCKET_PATH = "/tmp/alpamayo_flashdrive.sock"
DEFAULT_TCP_PORT = 5555
MAX_NEW_TOKENS = 128
DIFFUSION_STEPS = 8
CACHE_STEPS = [3, 4, 5, 6]


def send_msg(sock: socket.socket, data: Any) -> None:
    """Send arbitrary python object prefixed with 4-byte big-endian length."""
    payload = pickle.dumps(data, protocol=5)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Any:
    """Receive arbitrary python object prefixed with 4-byte big-endian length."""
    raw_len = sock.recv(4)
    if not raw_len:
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    chunks = []
    bytes_recd = 0
    while bytes_recd < msg_len:
        chunk = sock.recv(min(msg_len - bytes_recd, 65536))
        if not chunk:
            raise ConnectionError("Socket closed while reading payload")
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return pickle.loads(b"".join(chunks))


class FlashDriveEngine:
    """Manages model weights and streaming inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dummy: bool = False,
        num_samples: int = 1,
        torch_compile: str | None = None,
        expert_checkpoint: str | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.dummy = dummy
        self.num_samples = num_samples
        self.step_idx = 0

        if self.dummy:
            logger.info("Initializing in DUMMY mode (no GPU weights loaded).")
            return

        import flashdrive
        from flashdrive.streaming import convert_to_streaming_window

        self.flashdrive = flashdrive
        self.convert_to_streaming_window = convert_to_streaming_window

        # FlashDrive Option C: If a custom/local checkpoint is provided, load the
        # 4-bit PARO quantized base model (~11 GB VRAM) and patch the trained
        # Action Expert weights (416 tensors) into it.
        expert_ckpt = expert_checkpoint
        load_path = model_path
        if expert_ckpt is None and not model_path.lower().startswith("z-lab/"):
            expert_ckpt = model_path
            load_path = "z-lab/Alpamayo-1.5-10B"
            logger.info(
                f"[Option C] Detected local fine-tuned checkpoint: {expert_ckpt}. "
                f"Loading optimized 4-bit base model '{load_path}' and patching Action Expert."
            )

        logger.info(f"Loading Alpamayo model from {load_path} onto {device} (torch_compile={torch_compile})...")
        self.model = flashdrive.from_pretrained(load_path, device=device, torch_compile=torch_compile)
        self.optimized = isinstance(self.model, flashdrive.FlashDriveMixin)
        logger.info(f"Base model loaded. Optimized FlashDrive: {self.optimized}")

        if expert_ckpt:
            import glob
            from safetensors.torch import load_file
            logger.info(f"Extracting fine-tuned Action Expert weights from {expert_ckpt}...")
            safetensor_files = sorted(glob.glob(f"{expert_ckpt}/*.safetensors"))
            if not safetensor_files:
                raise FileNotFoundError(f"No .safetensors files found in {expert_ckpt}")
            expert_weights = {}
            for filepath in safetensor_files:
                shard = load_file(filepath, device="cpu")
                for key, tensor in shard.items():
                    if any(module in key for module in ["action", "expert", "diffusion", "delta"]):
                        expert_weights[key] = tensor.to(device=self.device, dtype=torch.bfloat16)

            missing, unexpected = self.model.load_state_dict(expert_weights, strict=False)
            mem_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(
                f"[Option C] Successfully patched {len(expert_weights)} fine-tuned expert tensors into base model! "
                f"Total VRAM allocated: {mem_gb:.2f} GB (Leaving plenty of headroom for CARLA)."
            )

        # Resolve helper package
        package = flashdrive.resolve_model_class(load_path).__module__.split(".")[0]
        self.helper = importlib.import_module(f"{package}.helper")
        self.camera_conditioned = "camera_indices" in inspect.signature(self.helper.create_message).parameters
        self.processor = self.helper.get_processor(self.model.tokenizer)

        self.vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        if self.optimized:
            self.rollout = functools.partial(
                self.model.sample_trajectories_streaming,
                max_new_tokens=MAX_NEW_TOKENS,
                diffusion_kwargs={
                    "inference_step": DIFFUSION_STEPS,
                    "cache_steps": CACHE_STEPS,
                    "int_method": "euler_with_cache",
                },
            )
        else:
            self.rollout = self.model.sample_trajectories_from_data_with_vlm_rollout

    def reset_stream(self) -> None:
        """Reset internal streaming step counter and KV cache."""
        self.step_idx = 0
        logger.info("Stream reset.")

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute one step of inference or generate dummy trajectory."""
        t_start = time.perf_counter()

        if self.dummy:
            # Generate plausible ego-coordinate future trajectory (x forward, y left)
            # Default horizon: 4 seconds at 10 Hz = 40 waypoints
            current_speed = float(request.get("speed", 5.0))
            cruise_speed = max(current_speed, 6.0)
            num_points = 40
            times = np.linspace(0.1, 4.0, num_points)
            # Straight path with slight heading variation
            pred_x = cruise_speed * times
            pred_y = np.zeros_like(pred_x)
            pred_z = np.zeros_like(pred_x)
            dummy_xyz = np.stack([pred_x, pred_y, pred_z], axis=-1)[None, :, :]  # (1, 40, 3)

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "status": "ok",
                "pred_xyz": dummy_xyz,
                "cot": "Dummy reasoning: maintain current lane and speed.",
                "latency_ms": elapsed_ms,
            }

        # Real model inference
        # request contains:
        # 'images': (N_cameras, H, W, C) uint8 or torch.Tensor
        # 'camera_indices': list of ints
        # 'ego_history_xyz': (T, 3)
        # 'ego_history_rot': (T, 3, 3)
        # 'nav_text': optional str
        images = request["images"]
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).permute(0, 3, 1, 2)  # (N, C, H, W) uint8
        elif isinstance(images, list):
            tensor_list = [torch.from_numpy(img).permute(2, 0, 1) for img in images]
            images = torch.stack(tensor_list, dim=0)

        logger.info(f"[step {self.step_idx}] images: {images.shape} dtype={images.dtype}")

        num_frames_per_camera = request.get("num_frames_per_camera", 4)
        camera_indices = torch.tensor(request["camera_indices"], dtype=torch.long)
        camera_kwargs = {"camera_indices": camera_indices} if self.camera_conditioned else {}
        nav_text = request.get("nav_text")

        logger.info(f"[step {self.step_idx}] camera_indices={camera_indices.tolist()}, "
                     f"num_frames_per_camera={num_frames_per_camera}, nav_text={nav_text!r}")

        create_msg_params = inspect.signature(self.helper.create_message).parameters
        create_kwargs = {}
        if "nav_text" in create_msg_params:
            create_kwargs["nav_text"] = nav_text
        if "num_frames_per_camera" in create_msg_params:
            create_kwargs["num_frames_per_camera"] = num_frames_per_camera
        if "camera_indices" in create_msg_params:
            create_kwargs["camera_indices"] = camera_indices

        messages = self.helper.create_message(images, **create_kwargs)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Debug: log tokenized shapes
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                logger.info(f"[step {self.step_idx}] tokenized_data[{k}]: {v.shape} dtype={v.dtype}")

        ego_hist_xyz = torch.tensor(request["ego_history_xyz"], dtype=torch.float32)
        while ego_hist_xyz.ndim < 4:
            ego_hist_xyz = ego_hist_xyz.unsqueeze(0)

        ego_hist_rot = torch.tensor(request["ego_history_rot"], dtype=torch.float32)
        while ego_hist_rot.ndim < 5:
            ego_hist_rot = ego_hist_rot.unsqueeze(0)

        # Defensive pad to 16 history timesteps (required for 48 history tokens)
        if ego_hist_xyz.shape[2] < 16:
            pad_len = 16 - ego_hist_xyz.shape[2]
            pad_xyz = ego_hist_xyz[:, :, :1, :].repeat(1, 1, pad_len, 1)
            ego_hist_xyz = torch.cat([pad_xyz, ego_hist_xyz], dim=2)
            pad_rot = ego_hist_rot[:, :, :1, :, :].repeat(1, 1, pad_len, 1, 1)
            ego_hist_rot = torch.cat([pad_rot, ego_hist_rot], dim=2)

        logger.info(f"[step {self.step_idx}] ego_hist_xyz: {ego_hist_xyz.shape}, ego_hist_rot: {ego_hist_rot.shape}")

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_hist_xyz,
            "ego_history_rot": ego_hist_rot,
        }

        if self.optimized and self.step_idx > 0:
            model_inputs = self.convert_to_streaming_window(
                model_inputs, self.vision_start_token_id, self.vision_end_token_id
            )

        data = self.helper.to_device(model_inputs, self.device)

        try:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.rollout(
                    data=data,
                    num_traj_samples=self.num_samples,
                    return_extra=True,
                )
        except RuntimeError as e:
            logger.error(f"[step {self.step_idx}] CUDA/runtime error during inference: {e}")
            # Try to recover GPU state
            torch.cuda.synchronize()
            self.step_idx += 1
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return {"status": "error", "message": str(e), "pred_xyz": None, "latency_ms": elapsed_ms}

        self.step_idx += 1
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if pred_xyz is None:
            # Prefill phase
            return {"status": "prefill", "pred_xyz": None, "latency_ms": elapsed_ms}

        cot = ""
        meta_action = ""
        if extra:
            if "cot" in extra:
                c = extra["cot"]
                if isinstance(c, np.ndarray):
                    cot = str(c.flat[0]) if c.size > 0 else ""
                elif isinstance(c, (list, tuple)) and len(c) > 0:
                    cot = str(c[0])
                else:
                    cot = str(c)
            if "meta_action" in extra:
                m = extra["meta_action"]
                if isinstance(m, np.ndarray):
                    meta_action = str(m.flat[0]) if m.size > 0 else ""
                elif isinstance(m, (list, tuple)) and len(m) > 0:
                    meta_action = str(m[0])
                else:
                    meta_action = str(m)

        if cot:
            logger.info(f"[step {self.step_idx}] latency={elapsed_ms:.1f}ms | CoC: {cot!r}")

        return {
            "status": "ok",
            "pred_xyz": pred_xyz.cpu().numpy(),
            "pred_rot": pred_rot.cpu().numpy() if pred_rot is not None else None,
            "cot": cot,
            "meta_action": meta_action,
            "latency_ms": elapsed_ms,
        }


def serve(args: argparse.Namespace) -> None:
    engine = FlashDriveEngine(
        model_path=args.model_path,
        device=args.device,
        dummy=args.dummy,
        num_samples=args.num_traj_samples,
        torch_compile=args.torch_compile,
        expert_checkpoint=getattr(args, "expert_checkpoint", None),
    )

    use_unix = args.use_unix
    if use_unix:
        sock_path = args.socket_path
        if os.path.exists(sock_path):
            os.remove(sock_path)
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(sock_path)
        logger.info(f"FlashDrive server listening on UNIX socket: {sock_path}")
    else:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((args.host, args.port))
        logger.info(f"FlashDrive server listening on TCP {args.host}:{args.port}")

    server_sock.listen(1)

    try:
        while True:
            logger.info("Waiting for client connection...")
            client_sock, _ = server_sock.accept()
            logger.info("Client connected.")
            engine.reset_stream()

            try:
                while True:
                    req = recv_msg(client_sock)
                    if req is None:
                        logger.info("Client disconnected.")
                        break

                    cmd = req.get("cmd", "step")
                    if cmd == "reset":
                        engine.reset_stream()
                        send_msg(client_sock, {"status": "reset_ok"})
                    elif cmd == "ping":
                        send_msg(client_sock, {"status": "pong"})
                    elif cmd == "step":
                        res = engine.step(req)
                        send_msg(client_sock, res)
                    else:
                        send_msg(client_sock, {"status": "error", "message": f"Unknown cmd: {cmd}"})
            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                logger.warning(f"Connection lost: {e}")
            finally:
                client_sock.close()
    finally:
        server_sock.close()
        if use_unix and os.path.exists(sock_path):
            os.remove(sock_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashDrive Alpamayo-1.5 Server")
    parser.add_argument("--model-path", default="z-lab/Alpamayo-1.5-10B", help="Model checkpoint path")
    parser.add_argument("--expert-checkpoint", default=None, help="Path to fine-tuned checkpoint to patch Action Expert")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--host", default="127.0.0.1", help="TCP host")
    parser.add_argument("--port", type=int, default=DEFAULT_TCP_PORT, help="TCP port")
    parser.add_argument("--socket-path", default=DEFAULT_SOCKET_PATH, help="Unix socket path")
    parser.add_argument("--use-unix", action="store_true", default=True, help="Use Unix domain socket")
    parser.add_argument("--tcp", action="store_false", dest="use_unix", help="Use TCP instead of Unix domain socket")
    parser.add_argument("--dummy", action="store_true", help="Run in mock/dummy mode for testing")
    parser.add_argument("--num-traj-samples", type=int, default=1, help="Number of trajectory samples")
    parser.add_argument(
        "--torch-compile",
        default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune"],
        help="Torch compile mode (default: None for optimal VRAM headroom)",
    )
    args = parser.parse_args()

    serve(args)


if __name__ == "__main__":
    main()
