# Fine-tuning pi05 (LoRA) on darmR pick-and-place

This folder is meant to be transferred whole to a RunPod pod for training. It contains
your LeRobot dataset **plus** the two openpi files needed to train on it:

| File | What it is | Goes to |
|---|---|---|
| `darm_policy.py` | Input/output transforms for this robot | `openpi/src/openpi/policies/darm_policy.py` |
| `darm_config_snippet.py` | Data config class + `TrainConfig` to paste | `openpi/src/openpi/training/config.py` |
| `README_TRAIN_DARM.md` | This file | — |

---

## ⚠️ BLOCKER — read first: your dataset is LeRobot **v3.0**, openpi needs **v2.1**

`meta/info.json` here says `"codebase_version": "v3.0"`. openpi (both this checkout and
current `main`) pins LeRobot to git rev `0cf8648…`, which is **`CODEBASE_VERSION = "v2.1"`**
and imports the old `lerobot.common.datasets` API. Its loader calls
`check_version_compatibility(...)`, which **raises** on a v3.0 dataset, and the v3.0
on-disk layout (`data/chunk-*/file-*.parquet`, `meta/episodes/`, `meta/tasks.parquet`)
is not what the v2.1 reader expects. **Training will not start until the dataset is v2.1.**

Updating openpi does **not** help — upstream `main` still pins the same v2.1 rev.

**Fix (built + verified):** regenerate the dataset in v2.1 from your aligned episodes with
`darm_align/to_lerobot21.py`, run under a v2.1 LeRobot. The easiest v2.1 LeRobot is
**openpi's own venv** — the exact lerobot that will read the dataset back:

```bash
cd /home/teleop/ann-mcap-aligned
./openpi/.venv/bin/python -m darm_align.to_lerobot21 \
    --aligned-dir /path/to/your/aligned_data \
    --root /mnt/ann-data/pnp/lerobot_datasets/darmR_pnp_both_v21 \
    --repo-id darmR_pnp_both --side both --no-tactile --vcodec h264 --overwrite
```

Use the **same flags** that produced the v3.0 build (`--side both --no-tactile`), and
`--vcodec h264` to match the existing dataset. This writes the v2.1 layout
(`data/chunk-000/episode_000000.parquet`, `videos/chunk-000/<key>/episode_000000.mp4`,
`meta/…`). It was smoke-tested end-to-end: openpi's loader reads the result correctly.
**Transfer `darmR_pnp_both_v21` to RunPod** (not the v3.0 dir) and use it in step 3 below.

> Note: this reads your **aligned episode folders** (the `arrays.npz`/`meta.json`/`images/`
> output of `darm_align.run`), not the v3.0 dataset. If those aligned folders are gone,
> ask me for the alternative v3.0→v2.1 in-place converter instead.

---

## 1. One-time pod setup (see the chat for GPU choice — 1× A100 80GB or H100 80GB)

```bash
cd /workspace                       # a RunPod *network volume*, so it survives restarts
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 2. Integrate the two files

```bash
cp /workspace/darmR_pnp_both/darm_policy.py src/openpi/policies/darm_policy.py
```
Then edit `src/openpi/training/config.py` and paste the three marked blocks from
`darm_config_snippet.py`:
- the `from openpi.policies import darm_policy` import (near the other policy imports),
- the `LeRobotDarmDataConfig` class (next to `LeRobotLiberoDataConfig`),
- the `TrainConfig(name="pi05_darm_pnp_lora", …)` entry inside the `_CONFIGS = [ … ]` list.

## 3. Place the dataset where LeRobot looks for it

openpi loads by `repo_id` only (no explicit path), resolving to
`$HF_LEROBOT_HOME/<repo_id>`. So:

```bash
export HF_LEROBOT_HOME=/workspace/lerobot_home
mkdir -p "$HF_LEROBOT_HOME"
# put the v2.1 dataset dir here as the repo_id used in the config:
ln -s /workspace/darmR_pnp_both "$HF_LEROBOT_HOME/darmR_pnp_both"
```
(The `repo_id` in the config is `darmR_pnp_both`. Keep the two in sync.)

## 4. Compute normalization stats (required — pi05 uses quantile norm)

```bash
cd /workspace/openpi
export HF_LEROBOT_HOME=/workspace/lerobot_home
export OPENPI_DATA_HOME=/workspace/openpi_cache
uv run scripts/compute_norm_stats.py --config-name pi05_darm_pnp_lora
```

## 5. Train (run inside tmux so an SSH drop doesn't kill it)

```bash
tmux new -s train
cd /workspace/openpi
export HF_LEROBOT_HOME=/workspace/lerobot_home
export OPENPI_DATA_HOME=/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9     # let JAX use 90% of VRAM
# export WANDB_API_KEY=...                     # or add --no-wandb below to disable

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_darm_pnp_lora \
    --exp-name=darm_pnp_lora_v1 --overwrite
```
Checkpoints land in `checkpoints/pi05_darm_pnp_lora/darm_pnp_lora_v1/<step>/`.
Resume after an interruption with `--resume` (instead of `--overwrite`).

## 6. Serve / inference after training

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_darm_pnp_lora \
    --policy.dir=checkpoints/pi05_darm_pnp_lora/darm_pnp_lora_v1/100000
```

---

## Config choices baked in (and how to tune them)

| Setting | Value | Why / how to change |
|---|---|---|
| Model | pi05, LoRA (`gemma_2b_lora` + `gemma_300m_lora`) | LoRA fits in >22.5 GB; freeze_filter + `ema_decay=None` are required for LoRA |
| `action_dim` | 32 | pi05 native; state(26)/action(28) auto-pad to 32 |
| `action_horizon` | 16 | ~0.53 s @ 30 fps. Raise to 25–32 for longer chunks (more open-loop, fewer inferences) |
| Actions | absolute (no delta) | darmR `cmd_*` are absolute joint targets. Delta variant is commented in the snippet |
| `discrete_state_input` | pi05 default (True) | matches the pi05 real-robot (DROID) finetune path |
| `num_train_steps` | 100_000 | as requested |
| `batch_size` | 32 | fits one 80GB card. **Value option:** batch 64 + 50k steps ≈ same training, ~20–30% cheaper — if you do this, roughly double `peak_lr` and halve `num_train_steps`/`decay_steps` |
| `peak_lr` | 2.5e-5 (cosine) | LoRA tolerates more; try up to 1e-4 if it underfits. Scale with batch size |
| `save_interval` / `keep_period` | 5k / 10k | 120 episodes × 100k steps ≈ ~90 epochs → **overfitting risk**; eval several checkpoints, don't assume step 100k is best |

## Dataset facts (from `meta/info.json`)
- 120 episodes, 78,379 frames, 30 fps, 1 task
- state 26-dim, action 28-dim (26 joints + Right_Hand + Left_Hand)
- cameras: head 720×1280, wrist_left 480×640, wrist_right 480×640 (all resized to 224×224 by the model transform)
