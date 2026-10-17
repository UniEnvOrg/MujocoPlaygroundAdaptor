"""Space metadata for MuJoCo Playground environments.

Playground only reports the sizes of its observations and actions; this module
describes the (single, unbatched) spaces with the shared unienv metadata
vocabulary (see ``unienv_interface.space.space_utils.metadata_utils``) from what
the environment and its config tell:

* a flat observation vector and the ``"state"`` entry of a dict observation are
  proprioceptive state, ``"privileged_state"`` is privileged (critic-only) state;
* ``"pixels/view_<i>"`` entries are camera images rendered with MJX-Warp
  (``mjx.get_rgb``: float RGB in ``[0, 1]``, ``(H, W, 3)``) at the control rate;
  the dm_control_suite cartpole stacks its last three grayscale frames
  (shifted by ``-0.5``) into the channel axis instead;
* actions are normalized to ``[-1, 1]``; they are named after the actuators when
  there is one action per actuator, and record what the actuators control and
  the config's ``action_scale``.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

import mujoco
import numpy as np

from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    image_metadata,
    metadata,
    proprio_metadata,
)

__all__ = ["playground_observation_metadata", "playground_action_metadata"]

_PIXELS_PREFIX = "pixels/view_"


def _config_value(env: Any, key: str) -> Any:
    config = getattr(env, "_config", None)
    if config is None:
        return None
    try:
        return config.get(key, None)
    except Exception:
        return None


def _json_number_or_list(value: Any) -> Any:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return [float(v) for v in array.reshape(-1)]


def _camera_name(env: Any, index: int) -> Optional[str]:
    try:
        model = env.mj_model
        if 0 <= index < model.ncam:
            return model.camera(index).name or None
    except Exception:
        pass
    return None


def _is_cartpole(env: Any) -> bool:
    return type(env).__module__.endswith("dm_control_suite.cartpole")


def _pixels_metadata(env: Any, key: str, shape: Tuple[int, ...]) -> Dict[str, Any]:
    view = key[len(_PIXELS_PREFIX):]
    source = _camera_name(env, int(view)) if view.isdigit() else None
    fps = 1.0 / float(env.dt)
    if _is_cartpole(env):
        # cartpole.py: gray = mean(rgb, -1) - 0.5, the last ``C`` frames stacked on the channel axis
        return image_metadata(
            format=None,
            layout="HWC",
            source=source,
            fps=fps,
            frame_stack=int(shape[-1]),
            label="stacked grayscale frames (oldest first), shifted by -0.5",
        )
    return image_metadata(
        format="rgb", layout="HWC", source=source, fps=fps, normalized=True, range=[0, 255]
    )


def _leaf_metadata(env: Any, key: Optional[str], shape: Tuple[int, ...]) -> Dict[str, Any]:
    if key is None or key == "state":
        return proprio_metadata()
    if key == "privileged_state":
        return metadata("privileged")
    if key.startswith(_PIXELS_PREFIX) and len(shape) == 3:
        return _pixels_metadata(env, key, shape)
    return {}


def playground_observation_metadata(
    env: Any, observation_size: Any
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """Metadata for the (unbatched) observation space of ``env``, for ``annotate``.

    ``observation_size`` is ``env.observation_size``: an ``int`` for a flat
    observation vector or a mapping from keys to shapes. Paths are tuples since
    Playground keys contain ``"/"`` (``"pixels/view_0"``).
    """
    if not isinstance(observation_size, Mapping):
        return {(): _leaf_metadata(env, None, (int(observation_size),))}
    result: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for key, size in observation_size.items():
        if isinstance(size, Mapping):  # no Playground env nests observations
            continue
        shape = (size,) if isinstance(size, int) else tuple(size)
        description = _leaf_metadata(env, key, shape)
        if description:
            result[(key,)] = description
    return result


def _actuator_control(model: mujoco.MjModel) -> Optional[str]:
    """``"joint_position"`` if all actuators are position servos on joints, ``"joint_torque"`` if all are motors."""
    if model.nu == 0 or not np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT):
        return None
    gain_fixed = model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED
    bias_none = model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE
    bias_affine = model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE
    # position servo: force = kp * ctrl - kp * qpos (- kv * qvel)
    kp = model.actuator_gainprm[:, 0]
    position = gain_fixed & bias_affine & np.isclose(model.actuator_biasprm[:, 1], -kp) & (kp > 0)
    motor = gain_fixed & bias_none
    if np.all(position):
        return "joint_position"
    if np.all(motor):
        return "joint_torque"
    return None


def playground_action_metadata(env: Any) -> Dict[str, Any]:
    """Metadata for the (unbatched) action space of ``env``: normalized to ``[-1, 1]``."""
    fields: Dict[str, Any] = {}
    model = env.mj_model
    if model.nu == env.action_size:
        fields["names"] = [model.actuator(i).name or str(i) for i in range(model.nu)]
        fields["control"] = _actuator_control(model)
        if "dm_control_suite" in type(env).__module__:
            # dm_control_suite envs write the action to the actuators as is
            ctrlrange = model.actuator_ctrlrange
            fields["range"] = [
                [float(v) for v in ctrlrange[:, 0]],
                [float(v) for v in ctrlrange[:, 1]],
            ]
    action_scale = _config_value(env, "action_scale")
    fields["action_scale"] = _json_number_or_list(action_scale)
    return action_metadata(normalized=True, **fields)
