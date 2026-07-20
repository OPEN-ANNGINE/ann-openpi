"""Policy input/output transforms for the darmR pick-and-place dataset (pi05).

╔══════════════════════════════════════════════════════════════════════════════╗
║ WORKED EXAMPLE — this file is filled in for the darmR pick-and-place robot.     ║
║ To train on YOUR dataset, copy it and change the lines marked `# 🔧 ADAPT:`.    ║
║ Everything else (image parsing, padding, train/infer plumbing) is generic and   ║
║ stays as-is. See README_TRAIN_DARM.md → "Adapt this to your dataset".            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Copy this file to `openpi/src/openpi/policies/darm_policy.py` on the training machine.

Dataset schema (from meta/info.json, LeRobot v3.0):
  observation.state          -> float32[26]  (7 L-arm + 7 R-arm + 6 L-fingers + 6 R-fingers)
  action                     -> float32[28]  (same 26 joints + Right_Hand + Left_Hand)
  observation.images.head    -> video 720x1280x3   -> base_0_rgb
  observation.images.wrist_left  -> video 480x640x3 -> left_wrist_0_rgb
  observation.images.wrist_right -> video 480x640x3 -> right_wrist_0_rgb

pi05 uses action_dim=32, so the 26-dim state and 28-dim action are zero-padded to 32
automatically by PadStatesAndActions in the model transforms -- do NOT pad here.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 🔧 ADAPT: number of real action dimensions in YOUR dataset (the width of the
# `action` column in meta/info.json). The rest is padding up to the model action_dim.
ACTION_DIM = 28


def make_darm_example() -> dict:
    """Random input example (used for smoke-testing the policy server)."""
    # 🔧 ADAPT: one key per camera your robot has + a `observation/state` of your
    # state width, and a representative `prompt`. The image shapes here are the raw
    # camera resolutions (before the model resizes to 224x224). Keys must match the
    # `observation/*` keys DarmInputs reads below.
    return {
        "observation/state": np.random.rand(26),
        "observation/head": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "observation/wrist_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the object and place it in the bin",
    }


def _parse_image(image) -> np.ndarray:
    """LeRobot serves video frames as float32 (C,H,W) in [0,1]; convert to uint8 (H,W,C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DarmInputs(transforms.DataTransformFn):
    """Convert a darmR sample into the model input format (training + inference)."""

    # Set by the data config from the model config; do not change.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 🔧 ADAPT: read one image per camera you have, and map each into the pi05
        # image slots. pi05 expects these fixed slot names: `base_0_rgb` (main/head),
        # `left_wrist_0_rgb`, `right_wrist_0_rgb`. If you have only one camera, use
        # `base_0_rgb` and drop the wrist slots (and their masks below); with two,
        # keep base + one wrist. The left-hand keys ("observation/head", …) are the
        # `observation/*` keys from make_darm_example / the RepackTransform.
        base_image = _parse_image(data["observation/head"])
        left_wrist_image = _parse_image(data["observation/wrist_left"])
        right_wrist_image = _parse_image(data["observation/wrist_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            # 🔧 ADAPT: one entry per slot above. np.True_ = camera always present.
            # For a slot you don't have, either omit it here and above, or (to keep a
            # fixed slot layout) feed a zero image and set its mask to np.False_.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Language instruction (populated from the LeRobot task via prompt_from_task=True).
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DarmOutputs(transforms.DataTransformFn):
    """Convert model output back to the darmR action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Strip the padding: keep only the first ACTION_DIM (28) action dimensions.
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
