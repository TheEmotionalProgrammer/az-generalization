import numpy as np
from environments.gridworld.grid_world import GridWorldEnv, grid_actions_dict
import gymnasium as gym

def observations_equal(obs1, obs2):

    """
    Compare two observations. 
    If needed, the method can be customized for different observation types.
    """

    if type(obs1) != type(obs2):
        return False
    else:
        if isinstance(obs1, (int, float)) and isinstance(obs2, (int, float)):
            return obs1 == obs2
        elif isinstance(obs1, np.ndarray) and isinstance(obs2, np.ndarray):
            return np.array_equal(obs1, obs2)
        else:
            raise NotImplementedError("Observation comparison not implemented for this type.")
    
def actions_dict(env: gym.Env) -> dict:

    """
    Returns the action dictionary for the given environment.
    """

    if isinstance(env.unwrapped, GridWorldEnv):
        return grid_actions_dict

    else:
        raise NotImplementedError(f"Unsupported environment: {env.spec.id}")
    
def print_obs(env: gym.Env, obs):

    """
    Prints the observation in a human-readable format.
    """

    if obs is None:
        return "None"

    elif isinstance(env.unwrapped, GridWorldEnv):
        ncols = env.unwrapped.ncol
        return obs // ncols, obs % ncols

    else:
        raise NotImplementedError(f"Unsupported environment: {env.spec.id}")