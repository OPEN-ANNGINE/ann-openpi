# Serve the darmR pick-and-place pi05 policy

Run the openpi WebSocket inference server for the LoRA-finetuned pi05 checkpoint
(`pi05_darm_pnp_lora`). This guide is written for someone starting **from a fresh
clone of this fork** — you only need this repo, `uv`, a GPU, and the checkpoint I
sent you a link to.

---

## What you need

| Requirement | Notes |
|---|---|
| OS | Linux |
| GPU | 1× NVIDIA GPU, **≥ 24 GB VRAM** (checkpoint restores to GPU; pi05 LoRA fits in ~22.5 GB) |
| Drivers | Recent NVIDIA driver + CUDA runtime (JAX GPU wheels are pulled in by the install) |
| Disk | ~30 GB free (repo + venv + checkpoint) |
| Tools | `git`, and `uv` (installed in step 2) |

You do **not** need the training dataset — normalization stats are baked into the
checkpoint.

---

## TL;DR

```bash
# 1. clone
git clone --recurse-submodules https://github.com/OPEN-ANNGINE/ann-openpi.git
cd ann-openpi

# 2. install (creates .venv)
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. download the checkpoint zip from the OneDrive link I sent (see step 3 for options),
#    then unzip it into checkpoints/
unzip darm_ckpt.zip -d checkpoints/      # -> checkpoints/30000/  (contains params/ + assets/)

# 4. serve
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_darm_pnp_lora \
  --policy.dir="checkpoints/30000"
```

Server is ready when you see `server listening on 0.0.0.0:8000`. Details for each
step below.

---

## 1. Clone the fork

```bash
git clone --recurse-submodules https://github.com/OPEN-ANNGINE/ann-openpi.git
cd ann-openpi
```

The DARM config (`pi05_darm_pnp_lora` in `src/openpi/training/config.py`) and the
policy transforms (`src/openpi/policies/darm_policy.py`) are committed to this fork,
so there is nothing to copy in by hand.

Quick check that the branch has them:

```bash
git log --oneline -1
grep -rq "pi05_darm_pnp_lora" src/openpi/training/config.py && echo "DARM config present ✅"
test -f src/openpi/policies/darm_policy.py && echo "DARM policy present ✅"
```

If either check fails, you are on a branch/commit without the DARM changes — ask me
which branch to use.

## 2. Install (one-time)

openpi uses [`uv`](https://docs.astral.sh/uv/) to manage its environment. This
creates a local `.venv/`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # skip if you already have uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` avoids pulling large LFS example assets you don't need for
serving. After this, run everything with `uv run …` (or call `.venv/bin/python`
directly).

Sanity-check that the config and policy import cleanly:

```bash
uv run python -c "
from openpi.training import config as c; print('config OK:', c.get_config('pi05_darm_pnp_lora').name)
from openpi.policies import darm_policy;   print('policy OK')"
```

## 3. Download the checkpoint (OneDrive)

I shared the checkpoint as a **zip on OneDrive**. Get it onto the machine one of two ways:

**A) Browser (simplest, always works):** open the OneDrive link, click **Download**,
then move the zip to the server (e.g. `scp darm_ckpt.zip user@server:~/ann-openpi/`).

**B) Headless / straight from the terminal:** OneDrive share links can be turned into
a direct download by appending `?download=1` (or `&download=1` if the link already has
a `?`):

```bash
curl -L "PASTE_THE_ONEDRIVE_LINK_HERE?download=1" -o darm_ckpt.zip
```

> OneDrive direct-download links are finicky (they redirect and sometimes need the
> `&download=1` form). If `curl` gives you an HTML page instead of a zip, use option A.

Then unzip it:

```bash
unzip darm_ckpt.zip -d checkpoints/
```

After unzipping you should have a **step directory** (`30000`) with this layout:

```
checkpoints/30000/
├── params/                # the model weights            <- required to serve
├── assets/                # baked-in normalization stats <- required to serve
│   └── darmR_pnp_both/norm_stats.json
├── train_state/           # optimizer state — only needed to *resume training*, ignored when serving
└── _CHECKPOINT_METADATA
```

That step directory (`checkpoints/30000` here) is what you pass as `--policy.dir`
below — it must be the directory that directly contains `params/` and `assets/`. If
your unzip nested it (e.g. `checkpoints/darm_pnp_lora_v1/30000/`), point `--policy.dir`
at that inner `30000` folder.

## 4. Run the server

```bash
CKPT=checkpoints/30000        # <- the step dir from step 3

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_darm_pnp_lora \
  --policy.dir="$CKPT"
```

- Startup takes ~25–30 s (restores params to GPU, fetches the PaliGemma tokenizer once).
- **Expected, not an error:** the log prints a line like
  `Norm stats not found in .../assets/pi05_darm_pnp_lora/..., skipping` and then loads
  them from the checkpoint's own `assets/darmR_pnp_both/norm_stats.json`.

You know it's ready when the log shows:

```
INFO:root:Creating server (host: ..., ip: ...)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### Keep it running past your SSH session

```bash
tmux new -s darm_serve
# run the serve command above, then detach with Ctrl-b then d
# reattach later:  tmux attach -t darm_serve
```

### Stop the server

`Ctrl-C` in the foreground, or from another shell:

```bash
kill "$(ss -ltnp 2>/dev/null | sed -n 's/.*:8000 .*pid=\([0-9]*\).*/\1/p' | head -1)"
```

## 5. Connect a client

The server is a WebSocket policy server on `0.0.0.0:8000`. From the robot side
(using `openpi-client`, which ships in `packages/openpi-client`):

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(host="<server-ip>", port=8000)

result = policy.infer({
    "observation/head":        head_img,        # uint8 HxWx3, 720x1280
    "observation/wrist_left":  wrist_left_img,  # uint8 480x640x3
    "observation/wrist_right": wrist_right_img, # uint8 480x640x3
    "observation/state":       state_26,        # float32[26] (7 L-arm + 7 R-arm + 6 L-fingers + 6 R-fingers)
    "prompt": "pick up the object and place it in the bin",
})
actions = result["actions"]   # shape [action_horizon=16, 28]
```

- Returned actions are **28-dim**: 26 joints + `Right_Hand` + `Left_Hand`, as
  **absolute joint targets** (no delta decoding needed).
- Images may be uint8 `HWC` **or** float `CHW` in `[0,1]` — the input transform
  handles both. State must be 26-dim (padded to 32 internally).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ValueError: Config 'pi05_darm_pnp_lora' not found` | You're on a branch without the DARM changes — see the check in step 1. |
| `ModuleNotFoundError: openpi.policies.darm_policy` | `src/openpi/policies/darm_policy.py` missing — same as above, re-clone the correct branch. |
| `CUDA out of memory` / XLA alloc error | GPU has < ~24 GB free, or another process holds VRAM (check `nvidia-smi`). Free it or use a bigger card. |
| `Address already in use` on :8000 | A server is already running — stop it (see above) or pass a different `--port`. |
| "Norm stats not found ... skipping" | **Expected** — see step 4. Stats load from the checkpoint. |
| Client `ConnectionRefused` | Server not up yet (wait for the "listening" log), wrong host/IP, or a firewall blocking the port. |

## Model facts (reference)

| Field | Value |
|---|---|
| config name | `pi05_darm_pnp_lora` |
| base | pi05 (`gs://openpi-assets/checkpoints/pi05_base`) |
| adapters | LoRA — `gemma_2b_lora` + `gemma_300m_lora` |
| action_dim | 32 (26 state / 28 action padded to 32) |
| action_horizon | 16 (~0.53 s @ 30 fps) |
| cameras | head 720×1280, wrist_left 480×640, wrist_right 480×640 (resized to 224×224 by the model) |
