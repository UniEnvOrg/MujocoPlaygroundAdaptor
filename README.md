# Mujoco Playground Adaptor

This repository contains the implementation of the adaptor that connects the Mujoco Playground environment to the UniEnv interface.

## Installation

You can install the adaptor from PyPI:

```bash
pip install unienv_playground
```

## Usage

By default our environment wrapper implements a `FuncEnv` interface, which explictly stores the environment state in user visible variables. If you want to use the more standard `Env` interface, you can use the `FuncEnvBasedEnv` wrapper provided by `unienv_interface` to convert it to a standard `Env`.

```python
from mujoco_playground import registry, MjxEnv
from unienv_playground import FromMJXPlaygroundEnv
from unienv_interface.env_base import FuncEnvBasedEnv, Env
import jax
import jax.numpy as jnp

num_envs = 128

mjxenv = registry.load(name)
funcenv = FromMJXPlaygroundEnv(
    mjxenv,
    num_envs,
    None, # Function for randomizing the environment, typed `Callable[[mjx.Model], Tuple[mjx.Model, AxisMapSingleT]]`, which takes in the original model and return the randomized model and the axis map for the randomized model. If None is provided, no randomization will be applied.
    jit=True # Whether to jit the environment step and reset function. Jitting can significantly improve the performance, but it also makes the first reset and step much slower due to the compilation time. We recommend setting it to True if you are going to run a long episode or run many episodes, and set it to False if you just want to quickly test the environment.
)
env = FuncEnvBasedEnv(
    funcenv,
)
ctx, obs, info = env.reset(seed=seed)
while True:
    action = env.sample_action()
    ctx, obs, reward, terminated, truncated, info = env.step(action)
    if jnp.any(terminated) or jnp.any(truncated):
        break
```

## Spaces

Actions are normalized to `[-1, 1]` (Playground's convention: dm_control_suite envs send them to actuators with ctrlrange `[-1, 1]`, locomotion and manipulation envs scale them by the config's `action_scale`). The spaces carry unienv space metadata: flat observations and `state` are `proprio`, `privileged_state` is `privileged`, `pixels/view_<i>` are `HWC` images at the control rate, and the action space records `normalized`, the actuator `names`, what the actuators control and the `action_scale`.
