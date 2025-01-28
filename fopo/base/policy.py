from typing import Any, Optional
from abc import ABC, abstractmethod

import numpy as np
import gymnasium as gym

class BasePolicy(ABC):
    """ Template class for policy. Initialized with an MDP environment
    instance and an optional seed number for randomly sampling.
    """
    
    def __init__(self, mdp: Any, discount_factor: float, seed: Optional[int]=None):
        self.mdp = mdp
        self.finite_action = isinstance(mdp.action_space, gym.spaces.Discrete) 
        self.finite_state = isinstance(mdp.observation_space, gym.spaces.Discrete)
        self.is_tabular = self.finite_action and self.finite_state

        self.discount_factor = discount_factor

        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, s: Any) -> Any:
        """ Given a state, samples an action according to current policy. """
        raise NotImplementedError

    @abstractmethod
    def policy_evaluate(self) -> Any:
        """ Evaluates current policy and returns output, which can be a lookup
        table in the tabular or parameters from function approximation.

        :returns output from policy evaluation (e.g., lookup table or function parameter)
        """
        # TODO: Return historical trajectory data for callbacks and logging
        raise NotImplementedError

    # @abstractmethod
    # def set_policy_spec(self, policy_spec: dict) -> None:
    #     """ Updates policy specification of policy, which can be the lookup 
    #     table for tabular MDP or parameters from function approximation and 
    #     the next step size.
    #     """
    #     raise NotImplementedError

