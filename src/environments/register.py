
import gymnasium as gym

def register():

    """
    Register all created custom environments that haven't been registered yet.
    """
      
    gym.register(
        id="GridWorldNoObst8x8-v1",
        entry_point="environments.gridworld.grid_world:GridWorldEnv",
        kwargs={
            "desc": [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFG"
                ],
            "map_name": None,
            "is_slippery": False,
            "terminate_on_obst": False,
            "hole_reward": 0,
        },
    )
