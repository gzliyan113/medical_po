from typing import Any, Optional, Tuple
from abc import ABC, abstractmethod

class MDP(ABC):
    """ Template class Markov decision process instance. The design of MDP
    follows closely to Farma Foundation's gymnasium
    (https://gymnasium.farama.org) with some modifications, such as:
        - Restarting at a specific state (instead of random initialization)
        - Enumerating all states and actions and explicit rewards and
          transition matrices for tabular MDPs. This is used in `utils.py`
          to compute metrics
        - The `step` function only returns next state, reward, and additional
          info (stored in a dictionary). 

    """

    @property
    def reward_matrix(self):
        """ Returns matrix of |S|x|A| size with rewards for tabular MDPs.
        Used in `utils.py` to compute metrics. For non-tabular MDPs, this 
        function will not be called and so it does not need to be implemented.
        does not need to be implemented.
        """
        raise NotImplementedError

    @property
    def transition_tensor(self):
        """ Returns 3D tensor of |S|x|S|x|A| size, where entry (s',s,a) is
        the probability of going s->s' with action a. Used in `utils.py` to
        compute metrics. For non-tabular MDPs, this function will not be called
        and so it does not need to be implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, a: Any) -> Tuple[Any, float, dict]:
        """ Performs since simulation step of the environment. First checks if 
        MDP has been reset (after forming the MDP or termination).

        :params a: action to take
        :returns next_s: next state
        :returns cost: cost for current (state, action)
        :returns info: additional info
        """
        pass
        
    @abstractmethod
    def reset(self, 
              seed: Optional[int]=None, 
              s_0: Optional[Any]=None) -> Tuple[Any, dict]:
        """ Resets environment, if allowed. If so, set `s_0` as the initial
        state. Otherwise if not allowed or no `s_0` is passed in, we use the 
        environment's default reset function (typically a random initial
        state).  We do not check whether state `s_0` is valid, so it is left to
        the user.

        :params s_0: initial state of system reset (if allowed)
        """
        pass


    @abstractmethod
    def sample_state(self) -> Any:
        """ 
        returned a ramdomly sampled state
        """
        pass


    @abstractmethod
    def sample_action(self) -> Any:
        """ 
        returned a ramdomly sampled action
        """
        pass
