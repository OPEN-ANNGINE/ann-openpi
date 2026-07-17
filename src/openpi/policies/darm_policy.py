"""Policy input/output transforms for the darmR pick-and-place dataset (pi05).

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

# Number of real action dimensions in the dataset (rest is padding up to model action_dim).
ACTION_DIM = 28


def make_darm_example() -> dict:
    """Random input example (used for smoke-testing the policy server)."""
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
            # All three cameras are real (present in every episode of this dataset).
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
