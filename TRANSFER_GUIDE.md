# FlashDrive: Deployment & Transfer Guide

This guide provides complete, step-by-step instructions for transferring, installing, and running the **FlashDrive** model server on a new machine or cloud instance.

---

## 1. System Requirements

- **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS recommended)
- **GPU**: NVIDIA GPU with Compute Capability 8.0+ (RTX 4090, RTX 6000 Ada, A100, H100) with **24 GB+ VRAM**
- **NVIDIA Driver**: Version 550+ (supports CUDA 12.4+)
- **System Memory**: 32 GB+ RAM recommended
- **Disk Storage**: At least 50 GB free disk space for base models and cached checkpoints
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (Astral) for lightning-fast Python dependency management

---

## 2. Quick Installation

### Step 1: Clone the Repository

```bash
git clone git@github.com:Mark-Teeratorn/flashdrive.git
cd flashdrive
```

*(Or via HTTPS: `git clone https://github.com/Mark-Teeratorn/flashdrive.git`)*

### Step 2: Set Up Python 3.12 Virtual Environment with `uv`

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create Python 3.12 virtual environment and sync dependencies
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Step 3: Configure CUDA Header Paths

FlashDrive relies on custom JIT-compiled CUDA kernels (e.g. ParoQuant rotation kernels). Ensure CUDA headers are discoverable in your environment:

```bash
# Add to ~/.bashrc or execute in your active terminal:
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

---

## 3. Model Architecture & Checkpoint Management

FlashDrive supports two execution workflows:

### A. Pre-trained Base Model (Zero-Shot)
The base model and companions are fetched automatically from Hugging Face:
- **Base model**: `z-lab/Alpamayo-1.5-10B`
- **W4A8 Quantized Companion**: `z-lab/Alpamayo-1.5-10B-PARO` (auto-downloaded)
- **DFlash Speculative Draft Model**: `z-lab/Alpamayo-1.5-10B-DFlash` (auto-downloaded)

### B. Fine-Tuned Action Expert Checkpoints (Option C)
During imitation learning (e.g., training on curated CARLA routes with LEAD), only the **Action Expert** weights (Diffusion denoiser transformer, `action_in_proj`, `action_out_proj`, and `delta_tokenizer`) are fine-tuned.

Raw training checkpoints are large (~21 GB for all shards). To save disk space and accelerate loading, extract the standalone Action Expert weights (~4.25 GB) using our extraction tool:

```bash
python scripts/extract_action_expert.py \
    --checkpoint /path/to/full_checkpoint_directory \
    --output-dir checkpoints/action_experts/checkpoint-XXXX
```

#### Extracted Directory Structure:
```
checkpoints/action_experts/checkpoint-XXXX/
├── action_expert.safetensors    # ~4.25 GB standalone bf16 weights (416 tensors)
├── config.json                  # Model configuration
└── metadata.json                # Tensor count and extraction timestamp
```

---

## 4. Running the FlashDrive Model Server

The server (`scripts/flashdrive_server.py`) hosts the 10B model and provides an ultra-low latency IPC bridge for driving clients (e.g. CARLA Leaderboard).

### Option 1: UNIX Domain Socket (Recommended for Local IPC)

Zero network stack overhead, fastest latency:

```bash
python scripts/flashdrive_server.py \
    --model-path checkpoints/action_experts/checkpoint-6400 \
    --socket-path /tmp/alpamayo_flashdrive.sock
```

### Option 2: TCP Socket (For Distributed / Networked Setups)

```bash
python scripts/flashdrive_server.py \
    --model-path checkpoints/action_experts/checkpoint-6400 \
    --tcp \
    --host 0.0.0.0 \
    --port 5555
```

### Option 3: Dummy / Mock Mode (For Testing Pipeline Without GPU)

```bash
python scripts/flashdrive_server.py --dummy --socket-path /tmp/alpamayo_flashdrive.sock
```

---

## 5. Verification & Health Check

Run a self-test in Python to confirm model weights load onto CUDA and patch cleanly:

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from flashdrive_server import FlashDriveEngine

engine = FlashDriveEngine(
    model_path='z-lab/Alpamayo-1.5-10B',
    expert_checkpoint='checkpoints/action_experts/checkpoint-6400',
    device='cuda',
    dummy=False,
    num_samples=1,
)
print('FlashDrive server engine loaded successfully!')
"
```

Expected log output:
```
[INFO] Loading Alpamayo model from z-lab/Alpamayo-1.5-10B onto cuda...
[INFO] Loading DFlash draft model from z-lab/Alpamayo-1.5-10B-DFlash
[INFO] Loading cached Action Expert directly from: .../action_expert.safetensors
[INFO] Loaded 416 expert tensors from cache in 0.82s!
[INFO] [Option C] Successfully patched 416 fine-tuned expert tensors into base model!
[INFO] Total VRAM allocated: 11.97 GB (Leaving plenty of headroom for CARLA).
```

---

## 6. IPC Protocol Reference

The server uses length-prefixed Python binary payloads:
- **Wire Format**: `[4 bytes big-endian length N] + [N bytes pickle payload]`

### Request Format
```python
{
    "cmd": "step",
    "images": np.ndarray,             # Shape (4, H, W, 3) uint8 or list of 4 frames
    "camera_indices": [0, 1, 2, 6],   # Cross-Left, Front-Wide, Cross-Right, Front-Tele
    "num_frames_per_camera": 4,       # History temporal window
    "nav_text": "Go straight along the road.", # Route guidance (6 tokens)
    "ego_history_xyz": np.ndarray,    # (16, 3) past positions
    "ego_history_rot": np.ndarray,    # (16, 3, 3) past rotation matrices
    "speed": 5.2,                     # Current speed in m/s
}
```

### Response Format
```python
{
    "status": "ok",
    "pred_xyz": np.ndarray,           # (1, 40, 3) future waypoints in ego frame (x forward, y left)
    "pred_rot": np.ndarray,           # (1, 40, 3, 3) future orientation
    "cot": str,                       # Generated Chain-of-Thought reasoning
    "meta_action": str,               # High-level tactical intention
    "latency_ms": float,              # End-to-end inference latency
}
```

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Missing quantization keys (252 keys)` | Normal ParoQuant rotation linear weights | Benign warning; quantized kernels handle these internally. |
| `CUDA out of memory during load` | CARLA or previous python process still holding VRAM | Kill dangling processes: `pkill -9 -f CarlaUE4; pkill -9 -f flashdrive_server.py`. |
| `Failed to compile paroquant_rotation.so` | Missing `cusolverDn.h` or CUDA dev headers | Ensure `export CUDA_HOME=/usr/local/cuda` and `sudo apt install cuda-toolkit-12-x`. |
| `Socket connection refused` | Server not finished loading | Model loading takes ~30–45s. Wait for `FlashDrive server listening on UNIX socket`. |
