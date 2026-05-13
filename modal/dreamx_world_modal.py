"""Modal runner for the TheMindExpansionNetwork DreamX-World fork.

This file keeps the fork synced to upstream while giving us a reproducible,
cache-backed smoke/inference lane on Modal. It intentionally uses persistent
volumes for HF/model caches and outputs so a failed/timeout run can resume
without re-downloading everything.

Usage from local Hermes checkout:

    cd /opt/data/workspace/projects/DreamX-World
    source /opt/data/hermes-agent/venv/bin/activate
    set -a; source /opt/data/.env; set +a
    modal run modal/dreamx_world_modal.py::precache
    modal run modal/dreamx_world_modal.py::run_three_samples --smoke true

For a higher quality run, bump frames/steps/resolution, e.g.:

    modal run modal/dreamx_world_modal.py::run_three_samples --smoke false --frames 81 --steps 30 --height 704 --width 1280
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import time
from typing import Iterable

import modal

APP_NAME = "dreamx-world-modal-toolbench"
UPSTREAM_REPO = "https://github.com/AMAP-ML/DreamX-World.git"
FORK_REPO = "https://github.com/TheMindExpansionNetwork/DreamX-World.git"
BASE_MODEL_REPO = "Wan-AI/Wan2.2-TI2V-5B"
DREAMX_MODEL_REPO = "GD-ML/DreamX-World-5B-Cam"

CACHE_ROOT = "/cache"
WORK_ROOT = "/workspace"
OUTPUT_ROOT = "/outputs"
REPO_DIR = f"{WORK_ROOT}/DreamX-World"
BASE_MODEL_DIR = f"{CACHE_ROOT}/models/Wan2.2-TI2V-5B"
DREAMX_MODEL_DIR = f"{CACHE_ROOT}/models/DreamX-World-5B-Cam"

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
dreamx_cache = modal.Volume.from_name("dreamx-world-cache", create_if_missing=True)
dreamx_outputs = modal.Volume.from_name("dreamx-world-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App(APP_NAME)

# Build notes:
# - CUDA 12.4 wheels match torch 2.5.1.
# - flash-attn is installed after torch with --no-build-isolation so it can find torch/CUDA.
# - We avoid baking model weights into the image; weights live in mounted Modal volumes.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "git-lfs", "ffmpeg", "libgl1", "libglib2.0-0", "ninja-build", "build-essential")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "Pillow",
        "einops",
        "safetensors",
        "timm",
        "tomesd",
        "librosa",
        "torchdiffeq",
        "torchsde",
        "decord",
        "datasets",
        "numpy<2",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
        "omegaconf",
        "SentencePiece",
        "albumentations",
        "imageio[ffmpeg]",
        "imageio[pyav]",
        "tensorboard",
        "beautifulsoup4",
        "ftfy",
        "func_timeout",
        "onnxruntime",
        "accelerate>=0.25.0",
        "gradio>=3.41.2",
        "diffusers>=0.30.1",
        "transformers>=4.46.2",
        "xfuser==0.4.1",
        "triton==3.1.0",
        "wcwidth==0.6.0",
        "huggingface_hub>=0.24.0",
        "hf_transfer",
    )
    .pip_install("wheel", "setuptools", "packaging")
    .run_commands("python -m pip install flash_attn==2.8.3 --no-build-isolation")
)


def _run(cmd: list[str] | str, cwd: str | None = None, env: dict | None = None) -> None:
    """Stream subprocess output live; do not use capture_output."""
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"command failed rc={rc}: {cmd}")


def _clone_or_update_repo() -> None:
    pathlib.Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(REPO_DIR, ".git").exists():
        _run(["git", "clone", FORK_REPO, REPO_DIR])
    _run(["git", "remote", "set-url", "origin", FORK_REPO], cwd=REPO_DIR)
    _run("git remote add upstream %s 2>/dev/null || git remote set-url upstream %s" % (UPSTREAM_REPO, UPSTREAM_REPO), cwd=REPO_DIR)
    _run("git remote set-url --push upstream DISABLED || true", cwd=REPO_DIR)
    _run(["git", "fetch", "origin", "--prune"], cwd=REPO_DIR)
    _run(["git", "fetch", "upstream", "--prune"], cwd=REPO_DIR)
    _run(["git", "checkout", "master"], cwd=REPO_DIR)
    _run(["git", "pull", "--ff-only", "origin", "master"], cwd=REPO_DIR)


def _hf_download(repo_id: str, local_dir: str) -> None:
    pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)
    _run(
        [
            "huggingface-cli",
            "download",
            repo_id,
            "--local-dir",
            local_dir,
            "--local-dir-use-symlinks",
            "False",
        ],
        env={"HF_HOME": f"{CACHE_ROOT}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"},
    )


@app.function(
    image=image,
    gpu=None,
    volumes={CACHE_ROOT: dreamx_cache, "/hf-cache": hf_cache},
    secrets=[hf_secret],
    timeout=6 * 60 * 60,
    startup_timeout=60 * 60,
)
def precache() -> dict:
    """Download upstream code and both model repos into persistent Modal volumes."""
    started = time.time()
    _clone_or_update_repo()
    _hf_download(BASE_MODEL_REPO, BASE_MODEL_DIR)
    _hf_download(DREAMX_MODEL_REPO, DREAMX_MODEL_DIR)
    dreamx_cache.commit()
    return {
        "ok": True,
        "status": "precache_complete",
        "base_model_dir": BASE_MODEL_DIR,
        "dreamx_model_dir": DREAMX_MODEL_DIR,
        "repo_dir": REPO_DIR,
        "elapsed_seconds": round(time.time() - started, 2),
    }


THREE_SAMPLE_ITEMS = [
    {
        "sample_id": "01_minecraft_coast",
        "image_path": "./demo/007.jpg",
        "caption": "Style: Minecraft. A peaceful blocky cliffside above a warm sunset ocean, with grassy terrain, flowers, sparse trees, pixelated clouds, and a slow forward camera move that lets the environment feel playable and explorable.",
        "action_seq": ["w", "wj"],
        "action_speed_list": [4, 6],
    },
    {
        "sample_id": "02_sci_fi_ice_ruins",
        "image_path": "./demo/005.png",
        "caption": "Style: 3D Rendering / CGI. A frozen alien sci-fi valley with luminous purple conduits under auroras, metallic ruins half buried in snow, and a camera that advances while tilting upward to reveal the scale of the world.",
        "action_seq": ["w", "wk"],
        "action_speed_list": [5, 6],
    },
    {
        "sample_id": "03_floating_sky_island",
        "image_path": "./demo/case6_04_天空之城巡游_shot_01_wl.png",
        "caption": "Style: 3D Rendering / CGI. A floating island kingdom above the clouds, with waterfalls, vine bridges, ancient arches, golden sunrise light, and a camera move that pushes forward while panning through the explorable path.",
        "action_seq": ["w", "wl", "wj"],
        "action_speed_list": [4, 6, 6],
    },
]


def _write_three_sample_json(path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(THREE_SAMPLE_ITEMS, ensure_ascii=False, indent=2), encoding="utf-8")


def _concat_grid(videos: Iterable[str], out_path: str) -> None:
    videos = list(videos)
    if len(videos) != 3:
        raise ValueError(f"expected three videos, got {len(videos)}")
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Normalize to 640x360 tiles, then stack horizontally.
    cmd = [
        "ffmpeg", "-y",
        "-i", videos[0], "-i", videos[1], "-i", videos[2],
        "-filter_complex",
        "[0:v]scale=640:360,setpts=PTS-STARTPTS[v0];"
        "[1:v]scale=640:360,setpts=PTS-STARTPTS[v1];"
        "[2:v]scale=640:360,setpts=PTS-STARTPTS[v2];"
        "[v0][v1][v2]hstack=inputs=3[v]",
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    _run(cmd)


@app.function(
    image=image,
    # The upstream script defaults to 8-way Ulysses. A single H100 smoke run uses CPU offload/qfloat8.
    # For production quality, switch to gpu="H100:8" and ulysses_degree=8 if budget allows.
    gpu="H100",
    volumes={CACHE_ROOT: dreamx_cache, OUTPUT_ROOT: dreamx_outputs, "/hf-cache": hf_cache},
    secrets=[hf_secret],
    timeout=10 * 60 * 60,
    startup_timeout=60 * 60,
)
def run_three_samples(
    smoke: bool = True,
    frames: int = 33,
    steps: int = 12,
    height: int = 384,
    width: int = 672,
    fps: int = 16,
    seed: int = 2045,
) -> dict:
    """Run three bounded DreamX-World samples and build a silent 3-up comparison video."""
    started = time.time()
    _clone_or_update_repo()
    if not pathlib.Path(BASE_MODEL_DIR).exists() or not any(pathlib.Path(BASE_MODEL_DIR).iterdir()):
        _hf_download(BASE_MODEL_REPO, BASE_MODEL_DIR)
    if not pathlib.Path(DREAMX_MODEL_DIR).exists() or not any(pathlib.Path(DREAMX_MODEL_DIR).iterdir()):
        _hf_download(DREAMX_MODEL_REPO, DREAMX_MODEL_DIR)

    run_slug = f"dreamx_three_env_{int(time.time())}_{'smoke' if smoke else 'quality'}"
    out_dir = f"{OUTPUT_ROOT}/{run_slug}"
    input_json = f"{REPO_DIR}/configs/dreamx/mindexpander_three_env_smoke.json"
    _write_three_sample_json(input_json)

    if smoke:
        frames = min(frames, 33)
        steps = min(steps, 12)
        height = min(height, 384)
        width = min(width, 672)

    cmd = [
        "python", "inference_dreamx5b.py",
        "--config_path", "configs/wan2.2/wan_ti2v_5b.yaml",
        "--model_name", BASE_MODEL_DIR,
        "--transformer_path", DREAMX_MODEL_DIR,
        "--input_dir", input_json,
        "--output_dir", out_dir,
        "--cam_method", "prope",
        "--add_control_adapter",
        "--sample_size", str(height), str(width),
        "--video_length", str(frames),
        "--fps", str(fps),
        "--guidance_scale", "3.0",
        "--num_inference_steps", str(steps),
        "--seed", str(seed),
        "--weight_dtype", "bfloat16",
        "--ulysses_degree", "1",
        "--ring_degree", "1",
        "--GPU_memory_mode", "model_cpu_offload_and_qfloat8" if smoke else "model_cpu_offload",
    ]
    _run(cmd, cwd=REPO_DIR, env={"HF_HOME": f"{CACHE_ROOT}/hf", "PYTHONUNBUFFERED": "1"})

    produced = sorted(str(p) for p in pathlib.Path(out_dir).glob("*.mp4"))
    grid_path = f"{out_dir}/dreamx_three_environment_grid_silent.mp4"
    if len(produced) >= 3:
        _concat_grid(produced[:3], grid_path)
        produced.append(grid_path)

    manifest = {
        "ok": True,
        "status": "three_samples_complete",
        "run_slug": run_slug,
        "output_dir": out_dir,
        "outputs": produced,
        "params": {"smoke": smoke, "frames": frames, "steps": steps, "height": height, "width": width, "fps": fps, "seed": seed},
        "elapsed_seconds": round(time.time() - started, 2),
    }
    pathlib.Path(out_dir, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dreamx_outputs.commit()
    dreamx_cache.commit()
    return manifest


@app.local_entrypoint()
def main(action: str = "precache", smoke: bool = True, frames: int = 33, steps: int = 12, height: int = 384, width: int = 672, fps: int = 16):
    if action == "precache":
        print(json.dumps(precache.remote(), indent=2))
    elif action == "run-three":
        print(json.dumps(run_three_samples.remote(smoke=smoke, frames=frames, steps=steps, height=height, width=width, fps=fps), indent=2))
    else:
        raise SystemExit(f"unknown action: {action}")
