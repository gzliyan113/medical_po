from fopo.base import BasePolicy

from typing import Any
from abc import ABC, abstractmethod

class BasePolicyOpt(ABC):
    """ Template class for policy optimization. Initialized with a policy
    instance.
    """
    
    def __init__(self, policy: BasePolicy):
        self.policy = policy
        self.is_tabular = policy.is_tabular

    # TODO: Rename learn() and train()
    def train(self, params) -> None:
        """ One iteration of a policy optimization algorithm """
        # TODO: Include logging and callbacks

        theta = self.policy.policy_evaluate()

        policy_spec = self.policy_update(theta, params)

        self.policy.set_policy_spec(policy_spec)
        
    @abstractmethod
    def learn(self):
        """ Multiple iterations of train to optimize policy. This is also 
        where we set the step size and callbacks before policy evaluation or
        policy update.
        """
        raise NotImplementedError

    @abstractmethod
    def policy_update(self, theta: Any, params: dict) -> dict:
        """ Performs a policy update. For tabular MDPs, it is a prox mapping. 
        In the non-tabular case, this returns the specifications for evalating
        the policy.

        :params theta: Output of policy evaluation
        :params params: Dictionary of algorithmic parameters (e.g., step size)
        :returns policy_spec: specification for next policy
        """
        raise NotImplementedError
