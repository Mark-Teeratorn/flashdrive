# FlashDrive

**Flash Vision-Language-Action Inference for Autonomous Driving**

[![Paper](https://img.shields.io/badge/arXiv-2608.12932-b31b1b.svg)](https://arxiv.org/abs/2608.12932)
[![Blog](https://img.shields.io/badge/Blog-FlashDrive-blue)](https://z-lab.ai/projects/flashdrive/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/collections/z-lab/flashdrive)

**FlashDrive** accelerates [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) and [Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B) — NVIDIA's 10B-parameter vision-language-action models for autonomous driving — by **4.5× with no loss in accuracy**, through algorithm-system co-design of five techniques, none sufficient alone:

- **Streaming inference** — reuse each frame's KV cache across windows, so a new window only prefills its latest frames.
- **Speculative reasoning** ([DFlash](https://github.com/z-lab/dflash)) — a block-diffusion draft proposes 8-token blocks that the target verifies in one forward, preserving its output distribution.
- **W4A8 quantization** ([ParoQuant](https://github.com/z-lab/paroquant)) — pairwise-rotation INT4 weights and INT8 activations via vLLM's Marlin kernels; the action expert stays bf16.
- **Adaptive action caching** — reuse the predicted velocity on select diffusion steps to skip action-expert forwards.
- **System optimizations** — static KV cache, fused expert projections, and `torch.compile` with CUDA graphs over the full decode loop.

## Performance

Measured with `scripts/infer.py` on a single RTX PRO 6000 over 100 PhysicalAI-AV
clips (one trajectory sample per step). minADE improves on both models.

| Model | minADE ↓ | Latency (per window) | Speedup |
|---|---|---|---|
| Alpamayo 1.5 | 1.705 | 717 ms | 1.0× |
| Alpamayo 1.5 + **FlashDrive** | **1.573** | **151 ms** | **4.7×** |
| Alpamayo 1 (R1) | 1.869 | 704 ms | 1.0× |
| Alpamayo 1 (R1) + **FlashDrive** | **1.662** | **155 ms** | **4.5×** |

Every stage of the rollout is accelerated (baseline → FlashDrive, ms):

| Stage | Alpamayo 1.5 | Alpamayo 1 (R1) |
|---|---|---|
| Encode | 87.0 → **12.0** (7.3×) | 86.4 → **12.0** (7.2×) |
| Prefill | 165.3 → **47.2** (3.5×) | 162.3 → **46.9** (3.5×) |
| Decode | 271.7 → **45.3** (6.0×) | 262.5 → **49.6** (5.3×) |
| Action | 192.9 → **46.9** (4.1×) | 192.6 → **46.9** (4.1×) |

## Quick Start

### Installation

Clone the repo and install with [uv](https://docs.astral.sh/uv/) (CUDA 12.8, Python 3.12, NVIDIA GPU with compute capability 8.0+):

```bash
git clone https://github.com/z-lab/flashdrive && cd flashdrive

uv venv --python 3.12
source .venv/bin/activate

uv sync
```

### Inference

Benchmark a PhysicalAI-AV clip — reports minADE and per-window latency:

```bash
python scripts/infer.py --model-path z-lab/Alpamayo-1.5-10B    # optimized
python scripts/infer.py --model-path nvidia/Alpamayo-1.5-10B   # baseline
```

A **z-lab** checkpoint runs the optimized stack; an **nvidia** checkpoint runs the original model.

### Library

```python
import flashdrive

model = flashdrive.from_pretrained("z-lab/Alpamayo-1.5-10B")

pred_xyz, pred_rot = model.sample_trajectories_streaming(data)
```

The first call per stream only prefills the KV cache and returns `(None, None)`; every later window returns trajectories. Passing an upstream `nvidia/...` checkpoint to `flashdrive.from_pretrained` loads the original model instead. See `scripts/infer.py` for building `data` from a PhysicalAI-AV clip.

## Models

All checkpoints are on the [Hugging Face collection](https://huggingface.co/collections/z-lab/flashdrive).
`flashdrive.from_pretrained` takes the **base** path; the W4A8 (`-PARO`) and DFlash
(`-DFlash`) companions are derived from it by suffix and fetched automatically.

| Model | Base | W4A8 ([ParoQuant](https://github.com/z-lab/paroquant)) | Draft ([DFlash](https://github.com/z-lab/dflash)) |
|---|---|---|---|
| Alpamayo 1.5 | [z-lab/Alpamayo-1.5-10B](https://huggingface.co/z-lab/Alpamayo-1.5-10B) | [-PARO](https://huggingface.co/z-lab/Alpamayo-1.5-10B-PARO) | [-DFlash](https://huggingface.co/z-lab/Alpamayo-1.5-10B-DFlash) |
| Alpamayo 1 (R1) | [z-lab/Alpamayo-R1-10B](https://huggingface.co/z-lab/Alpamayo-R1-10B) | [-PARO](https://huggingface.co/z-lab/Alpamayo-R1-10B-PARO) | [-DFlash](https://huggingface.co/z-lab/Alpamayo-R1-10B-DFlash) |

## Autonomous Driving Model Server (LEAD / CARLA Integration)

FlashDrive includes an ultra-low latency IPC model server for closed-loop evaluation in simulators such as CARLA 0.9.15/0.9.16 via the **LEAD** framework.

### 1. Extracting Fine-Tuned Action Expert Weights
When fine-tuning on custom datasets (e.g. curated routes), only the Action Expert weights are updated. To avoid storing and loading full 21 GB sharded checkpoints, use our extractor:

```bash
python scripts/extract_action_expert.py \
    --checkpoint /path/to/checkpoint-XXXX \
    --output-dir checkpoints/action_experts/checkpoint-XXXX
```
This extracts the 416 diffusion and projection tensors into a standalone `action_expert.safetensors` (~4.25 GB).

### 2. Starting the FlashDrive Server
Launch the server listening on a UNIX domain socket (zero network latency):

```bash
python scripts/flashdrive_server.py \
    --model-path checkpoints/action_experts/checkpoint-6400 \
    --socket-path /tmp/alpamayo_flashdrive.sock
```

The server automatically loads the W4A8 quantized 10B base model (~11.9 GB VRAM) and patches the fine-tuned action expert on the fly, leaving over 12 GB of VRAM headroom for the simulator on consumer GPUs like the RTX 4090.

> [!TIP]
> For complete instructions on transferring and deploying FlashDrive to a new server or cloud VM, see [TRANSFER_GUIDE.md](TRANSFER_GUIDE.md).

## License

FlashDrive's code is released under the [MIT License](LICENSE). The Alpamayo weights
mirror NVIDIA's release and remain under its
[non-commercial license](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/main/LICENSE).

## Citation

```bibtex
@inproceedings{li2026flashdrive,
  title     = {{FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving}},
  author    = {Li, Zekai and Liang, Yihao and Zhang, Hongfei and Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2608.12932}
}
```

FlashDrive builds on DFlash and ParoQuant; please consider citing them as well:

```bibtex
@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}

@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Zhang, Zihan and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

