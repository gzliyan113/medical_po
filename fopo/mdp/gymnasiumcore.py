from typing import Any, Optional, Tuple
from fopo.mdp import MDP
from fopo.base import utils

import numpy as np

from gymnasium.wrappers import TransformReward, TransformObservation

class GymnasiumTaxi(MDP):
    """ Instance of simple Taxi problem
    (https://gymnasium.farama.org/environments/toy_text/taxi/). State space is 
    of size 500 while action is 6. We build our own instance on top of
    gymnasium so that the states and actions are returned as tuples, and we can
    reset to particular states.

    starting state: randomly selected --> performance eval should be adapted to how the initial state is set
    """
    def __init__(self, discount_factor, **kwargs):

        env = utils.setup_gymnasium_env('Taxi-v3', **kwargs)
        self.env = env

        self.states = np.arange(
            env.observation_space.start, 
            env.observation_space.start+env.observation_space.n
        )
        self.state_dim = 1

        self.actions = np.arange(
            env.action_space.start, 
            env.action_space.start+env.action_space.n
        )
        self.action_dim = 1

        self.is_tabular = True
        self.discount_factor = discount_factor

    def step(self, a: Any) -> Tuple[Any, float, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(a)
        # Save `terminated` and `truncated` in `info` (dict)
        info['terminated'] = terminated
        info['truncated'] = truncated

        # # reset to the initial state if the terminated or truncated
        # if terminated or truncated:
        #     next_state, _ = self.env.reset()

        # do not truncate, otherwise transit kernel depend on timestep (need to make sure truncation is removed
        # perhaps check gym source code: https://github.com/openai/gym/blob/master/gym/envs/__init__.py#L122
        # )
        if terminated: 
            next_state, _ = self.env.reset()

        return next_state, -reward, info

    def reset(self, 
              seed: Optional[int]=None,
              s_0: Optional[Any]=None) -> Tuple[Any, dict]:

        # taxi-v3 does not take in a seed during reset
        state, info = self.env.reset()

        if s_0 is not None:
            # hack to force a state (found by manual testing)
            # self.env.env.env.env.s = s_0
            base_env = self.env.unwrapped
            base_env.s = s_0
            state = s_0

        return state, info
    
    def sample_state(self) -> Any:
        return np.random.choice(np.array(self.states))
    
    def sample_action(self) -> Any:
        return np.random.choice(np.array(self.actions))

class MountainCar(MDP):
    """ Instance of Mountain Car, which is continuous space finite action
    (https://gymnasium.farama.org/environments/classic_control/mountain_car/).
    State space a 2D region (Box([-1.2, -0.07], [0.6, 0.07])), corresponding to
    the position (along the x-axis) and velocity of the car. The action space
    is \{0,1,2\}, corresponding to accelerate left, don't accelerate, and
    accelerate right. More details can be found in the link provided above.

    If you do not want truncation (Mountain-Car automatically truncates after
    200 steps), pass in the extra parameter `no_truncation` and set it to True.

    starting state: random position, 0 speed --> performance eval should be adapted to how the initial state is set
    """
    def __init__(self, discount_factor, **kwargs):

        env = utils.setup_gymnasium_env('MountainCar-v0', **kwargs)
        self.env = env

        self.states = env.observation_space
        self.state_dim = 2

        self.actions = np.arange(
            env.action_space.start, 
            env.action_space.start+env.action_space.n
        )
        self.action_dim = 1

        self.is_tabular = False
        self.discount_factor = discount_factor
            
    def step(self, a: Any) -> Tuple[Any, float, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(a)

        # Save `terminated` and `truncated` in `info` (dict)
        info['terminated'] = terminated
        info['truncated'] = truncated

        # reset to the initial state if the terminated or truncated
        if terminated:
            # MountainCar has no explicit reward for terminating, we do it ourselves
            reward = 100.
            next_state, _ = self.reset()
        elif truncated:
            pass

        return next_state, -reward, info

    def reset(self, 
              seed: Optional[int]=None,
              s_0: Optional[np.ndarray]=None) -> Tuple[Any, dict]:

        state, info = self.env.reset(seed=seed)

        if s_0 is not None:
            # hack to force a state in taxi-v3 (found by manual testing)
            for i in range(self._dim):
                assert self._low[i] <= s_0[i] <= self._high[i], "Reset state not within allowable dimensions"
            base_env = self.env.unwrapped
            base_env.state = s_0
            state = s_0
        else:
            s_0 = state
            rng = np.random.default_rng(seed)
            # only change init position. 
            s_0[0] = rng.uniform(self._low[0], self._high[0])
            base_env = self.env.unwrapped
            base_env.state = s_0
            state = s_0

        return state, info

    def sample_state(self) -> Any:
        return self.states.sample()
    
    def sample_action(self) -> Any:
        return np.random.choice(np.array(self.actions))

    
class LunarLander(MDP):
    """ Instance of Lunar Lander, which is continuous space finite action
    (https://gymnasium.farama.org/environments/box2d/lunar_lander/).
    State space a 8D region, corresponding to (in the listed order):
        - lander (x,y) coordinates
        - lander linear (x,y) velocities
        - lander angle
        - lander angular velocity
        - two booleans for each of the (two) legs if touching the ground
    More specifically:
        Box([-1.5, -1.5, -5, -5, -3.1415297, -5, 0, 0],
            [ 1.5,  1.5,  5,  5,  3.1415297,  5, 1, 1])
    Action space is \{0,1,2,3\}, corresponding to do nothing, fire left
    orientation engine, fire main engine, and fire right orientation engine.

    If you do not want truncation (Mountain-Car automatically truncates after
    200 steps), pass in the extra parameter `no_truncation` and set it to True.

    starting state: fix center position, random initial force
    """
    def __init__(self, discount_factor, **kwargs):
        
        env = utils.setup_gymnasium_env("LunarLander-v2", **kwargs)

        if True:
            # Scale reward so approximately in [0,1]
            # env = TransformReward(env, lambda r: 0.1*r)
            # Scale observation so approximately between [-1,1]
            d = np.array([0, 0, 0, 0, 0, 0, 0.5, 0.5])
            D = np.diag(np.reciprocal([1.5, 1.5, 5, 5, 3.1415297, 5, 0.5, 0.5]))
            env = TransformObservation(env, lambda s: np.dot(D, s-d))

        self.env = env

        self.states = env.observation_space
        self.state_dim = 8

        self.actions = np.arange(
            env.action_space.start, 
            env.action_space.start+env.action_space.n
        )
        self.action_dim = 1

        self.is_tabular = False
        self.discount_factor = discount_factor

    def step(self, a: Any) -> Tuple[Any, float, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(a)

        # Save `terminated` and `truncated` in `info` (dict)
        info['terminated'] = terminated
        info['truncated'] = truncated

        # reset to the initial state if the terminated or truncated
        # the loss seems to be very large if reset is not implemented -- this
        # is likely due to that if the lander crash and we do not reset, then
        # high cost is incurred every round starting from the moment the lander
        # crashes I did not see the vanilla implementation in Gym
        # https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py
        # handling reset to initial state explicitly, perhaps their method
        # itself is episodic in nature, while our method (in particular policy
        # eval) is infinite horizon note that termination criterion is made
        # based on status of the next state
        # TODO: again need to remove truncation 
        if terminated:
            next_state, _ = self.env.reset()

        return next_state, -reward, info

    def reset(self, 
              seed: Optional[int]=None,
              s_0: Optional[np.ndarray]=None) -> Tuple[Any, dict]:

        state, info = self.env.reset(seed=seed)

        return state, info

    def sample_state(self) -> Any:
        return self.states.sample()

    def sample_action(self) -> Any:
        return np.random.choice(np.array(self.actions))
    
class CartPole(MDP):
    """ Instance of Cart Pole
    (https://gymnasium.farama.org/environments/classic_control/cart_pole/).
    State space a 4D region, corresponding to (in the listed order):
        - cart position (Box[-4.8, 4.8])
        - cart velocity (Box[-inf, inf])
        - pole angle (Box[-0.418, 0.418])
        - pole ang vel (Box[-inf, inf])
    Action space is \{0,1\}, for moving left and right, respectively.
    """
    def __init__(self, discount_factor, **kwargs):

        env = utils.setup_gymnasium_env("CartPole-v1", **kwargs)
        self.env = env

        self.states = env.observation_space
        self.state_dim = 4

        self.actions = np.arange(
            env.action_space.start, 
            env.action_space.start+env.action_space.n
        )
        self.action_dim = 1

        self.is_tabular = False
        self.discount_factor = discount_factor

    def step(self, a: Any) -> Tuple[Any, float, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(a)

        # Save `terminated` and `truncated` in `info` (dict)
        info['terminated'] = terminated
        info['truncated'] = truncated

        if terminated or truncated:
            next_state, _ = self.env.reset()

        return next_state, -reward, info

    def reset(self, 
              seed: Optional[int]=None,
              s_0: Optional[np.ndarray]=None) -> Tuple[Any, dict]:

        state, info = self.env.reset(seed=seed)

        return state, info

    def sample_state(self) -> Any:
            return self.states.sample()
    
    def sample_action(self) -> Any:
        return np.random.choice(np.array(self.actions))