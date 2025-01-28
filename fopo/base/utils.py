from typing import Any
import numpy as np
import gymnasium as gym

def modify_reset(env: gym.Env) -> gym.Env:
    """ Modifies env.reset() to allow reset at a specific state. We override the
        `reset` function with a custom one. Requires env supports the following
        functions:
            - env.unwrapped
            - env.reset(...)
            - env.s (for base environment)

        Currently, some wrappers 
        (e.g. https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/time_limit.py)
        fail when a seed is passed in. 

        TODO: Pass in seed with TimeLimit (file bug report) or handle it internally

    """
    gym_reset_name = "gymnasium_reset"
    reset_unmodified = gym_reset_name not in dir(env)

    if reset_unmodified:
        base_env_type = type(env.unwrapped) 
        setattr(base_env_type, gym_reset_name, base_env_type.reset)

        # def custom_reset(self, seed = None, options: dict[str, Any] = None) -> tuple[gym.ObsType, dict[str, Any]]:
        def custom_reset(self, seed = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
            """ Enhances gymnasium's reset
            (https://gymnasium.farama.org/api/env/#gymnasium.Env.reset) by
            checking `options` for an initial state (with key `s_0`).  Validity
            of state is not checked.
                
            :params seed: The seed that is used to initialize the environment's PRNG.
            :params options: Additional information to specify how the environment is reset 
            """
            
            (state, info) = self.gymnasium_reset() 

            if options is not None and options.get("s_0", None) is not None:
                s_0 = options["s_0"]
                # base_env = self.env.unwrapped
                # base_env = self.env
                # base_env.s = s_0
                # The lowest level might be calling
                self.s = s_0
                state = s_0

            return (state, info)

        # https://block.arch.ethz.ch/blog/2016/07/adding-methods-to-python-classes/
        setattr(base_env_type, "reset", custom_reset)

    return env

def convert_discrete_to_array(space) -> np.ndarray:
    """ Converts Gymnasium Discrete space to numpy array 

    :params space: instance of discrete space
    :returns: numpy array of space
    """
    if not isinstance(space, gym.spaces.Discrete):
        print(f"Input type {type(space)}, but required to be `gym.spaces.Discrete`")
        return None

    return np.arange(space.start, space.start+space.n)

def is_finite_space(space) -> bool:
    """ Returns true/false if space is tabular """
    return isinstance(space, gym.spaces.Discrete)

def setup_gymnasium_env(env_name: str, **kwargs) -> gym.Env:
    """ Setups basic gymnasium environment 

    :params env_name: ID of gymnasium environment
    :params **kwargs: optional features, including
        - max_episode_steps: change max episode length
        - animate: whether to animate the environment
    :returns env: Gymnasium environment
    """
    # TODO: Might need arguments to pass into gymnasium

    animate = False
    max_episode_steps = -1
    for key,val in kwargs.items():
        if key == "max_episode_steps" and val:
            max_episode_steps = int(val)
        elif key == "animate" and val:
            animate = True

    if animate:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    if max_episode_steps > 0:
        env._max_episode_steps = max_episode_steps

    return env
