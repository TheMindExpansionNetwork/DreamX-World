# MindExpander DreamX-World Modal Toolbench

This fork tracks upstream `AMAP-ML/DreamX-World` while keeping Jimsky/MindExpander operational glue in the fork.

## Remotes

- `origin`: `https://github.com/TheMindExpansionNetwork/DreamX-World.git`
- `upstream`: `https://github.com/AMAP-ML/DreamX-World.git`
- `upstream` push URL is intentionally set to `DISABLED` locally so automation cannot accidentally push to the source project.

## Sync upstream safely

```bash
git fetch upstream --prune
git checkout master
git merge --ff-only upstream/master
git push origin master
```

If upstream changes conflict with local Modal tooling, create a branch and PR instead of force-pushing.

## Modal workflow

The Modal runner lives at:

```text
modal/dreamx_world_modal.py
```

It uses persistent Modal volumes:

- `dreamx-world-cache`: Hugging Face model snapshots and working repo cache.
- `dreamx-world-outputs`: generated DreamX sample videos and manifests.
- `huggingface-cache`: shared HF cache mount.

Required Modal secret:

- `huggingface-secret` with `HF_TOKEN` / `HUGGINGFACE_TOKEN` available.

## Commands

From this repo on the Hermes host:

```bash
cd /opt/data/workspace/projects/DreamX-World
source /opt/data/hermes-agent/venv/bin/activate
set -a; source /opt/data/.env; set +a

# Pre-cache upstream repo and model weights into Modal volumes.
modal run modal/dreamx_world_modal.py --action precache

# Run a bounded three-environment smoke test.
modal run modal/dreamx_world_modal.py --action run-three --smoke true

# Higher-quality sample pass once smoke is good.
modal run modal/dreamx_world_modal.py --action run-three --smoke false --frames 81 --steps 30 --height 704 --width 1280 --fps 16
```

## Three environment smoke pack

The runner creates three test lanes:

1. Minecraft sunset coast / playable block-world navigation.
2. Frozen sci-fi alien ruins / aurora environment.
3. Floating sky island / fantasy exploration path.

It then combines the three MP4s into a silent 3-up comparison grid:

```text
/outputs/<run_slug>/dreamx_three_environment_grid_silent.mp4
```

The next packaging step is to pull the grid locally, add MindExpander voice narration, and render the final instructional video.

## Cost / safety defaults

- Smoke defaults are intentionally tiny: `33` frames, `12` steps, `384x672`, single `H100`, CPU-offload + qfloat8.
- Full upstream quality can be expensive. Use `H100:8` / Ulysses 8 only when explicitly desired.
- Do not commit model weights, generated videos, Modal logs, `.env`, tokens, or personal voice artifacts.
