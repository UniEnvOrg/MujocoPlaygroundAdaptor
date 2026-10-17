import jax.numpy as jnp
import numpy as np
from utils import construct_env_from_name, perform_env_test
import pytest

@pytest.mark.parametrize("env_name", ["CartpoleBalance"])
@pytest.mark.parametrize("n_envs", [16, 32])
@pytest.mark.parametrize("seed", [0, 1024])
@pytest.mark.parametrize("jit", [True]) # If not traced it will be tooooooo slow
def test_dmsuite(
    env_name : str,
    n_envs : int,
    seed : int,
    jit : bool,
    episodes : int = 5,
    max_steps : int = 200
):
    env = construct_env_from_name(env_name, n_envs, jit, seed=seed)
    perform_env_test(env, episodes, max_steps)
    env.close()

def test_action_space_is_normalized():
    # Playground actions live in [-1, 1] (for CartpoleBalance they are written to
    # the slider actuator, whose ctrlrange is [-1, 1]); the space used to be unbounded.
    env = construct_env_from_name("CartpoleBalance", 4, True, seed=0)
    single_env = env.func_env.single_env
    assert env.action_space.shape == (4, single_env.action_size)
    assert np.all(np.asarray(env.action_space.low) == -1.0)
    assert np.all(np.asarray(env.action_space.high) == 1.0)
    assert np.all(single_env.mj_model.actuator_ctrlrange == np.array([-1.0, 1.0]))

    action = env.sample_action()
    assert action in env.action_space
    assert np.all(np.abs(np.asarray(action)) <= 1.0)
    assert jnp.full((4, single_env.action_size), 2.0) not in env.action_space
    env.close()
