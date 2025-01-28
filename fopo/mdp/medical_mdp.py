
from typing import Any, Optional, Tuple
from fopo.mdp import MDP
import numpy as np
from fopo.mdp.nn_model import SimpleResNet, SimpleLinearModel, ResidualBlock
import torch


class MedicalMDP(MDP):
    def __init__(self, **kwargs):
        """
        kwargs[states]: list of ranges for each state var 
        kwargs[actions]: list of ranges for each action var 
        kwargs[num_discretizations]: list of number discretization for each action var
        """
        self.nn = None         
        self.is_tabular = False

        self.states_range = kwargs['states']
        self.actions_range = kwargs['actions']
        self.num_discretizations = kwargs['num_discretizations']

        self.states = self.states_range 
        self.actions = np.arange(self.get_total_actions()) 

        self.observation_space, self.action_space = None, None 

        # set initial state 
        self.state, _ = self.reset(kwargs['initial_state']) if 'initial_state' in kwargs else self.sample_state()

    def flatten_index(self, indices):
        """
        Flattens a multidimensional index into a single integer.
        """
        flat_index = 0
        multiplier = 1
        for i, num_discretization in zip(reversed(indices), reversed(self.num_discretizations)):
            if num_discretization > 1:  # Skip if num_discretization is 1
                flat_index += i * multiplier
                multiplier *= num_discretization
        return flat_index    
    
    def unflatten_index(self, flat_index):
        """
        Converts a flattened index back into multidimensional indices.
        """
        indices = []
        for num_discretization in reversed(self.num_discretizations):
            if num_discretization > 1:  # Skip if num_discretization is 1
                indices.append(flat_index % num_discretization)
                flat_index //= num_discretization
            else:
                indices.append(0)  # Default to 0 for degenerate dimensions
        return tuple(reversed(indices))
            
    def encode_action_to_integer(self, action_values):
        """
        Encodes the given action values into a single integer.
        """
        discretized_actions = []
        for j, (min_val, max_val) in enumerate(self.actions_range):
            if self.num_discretizations[j] > 1:
                # Discretize the action range if num_discretizations > 1
                discretized_actions.append([
                    min_val + i * (max_val - min_val) / (self.num_discretizations[j] - 1)
                    for i in range(self.num_discretizations[j])
                ])
            else:
                # If num_discretizations is 1, use the min_val directly
                discretized_actions.append([min_val])

        # Find the closest indices for each action value
        action_indices = []
        for j, action_value in enumerate(action_values):
            if self.num_discretizations[j] > 1:
                # Find the closest index if num_discretizations > 1
                index = min(range(self.num_discretizations[j]), 
                            key=lambda i: abs(action_value - discretized_actions[j][i]))
                action_indices.append(index)
            else:
                # If num_discretizations is 1, default to index 0
                action_indices.append(0)

        # Flatten the indices into a single integer
        return self.flatten_index(tuple(action_indices))    
        
    def decode_action_from_integer(self, action_integer):
        """
        Decodes the given integer into action values.
        """
        # Unflatten the integer to multidimensional indices
        action_indices = self.unflatten_index(action_integer)

        # Discretize each action range
        discretized_actions = []
        for j, (min_val, max_val) in enumerate(self.actions_range):
            if self.num_discretizations[j] > 1:
                # Discretize the action range if num_discretizations > 1
                discretized_actions.append([
                    min_val + i * (max_val - min_val) / (self.num_discretizations[j] - 1)
                    for i in range(self.num_discretizations[j])
                ])
            else:
                # If num_discretizations is 1, use the min_val directly
                discretized_actions.append([min_val])

        # Map indices to actual action values
        action_values = []
        for j, index in enumerate(action_indices):
            if self.num_discretizations[j] > 1:
                # Use the discretized action value if num_discretizations > 1
                action_values.append(discretized_actions[j][index])
            else:
                # If num_discretizations is 1, use the min_val directly
                action_values.append(discretized_actions[j][0])

        return np.array(action_values)

    def step(self, a):
        # decode action a into concrete values
        a = self.decode_action_from_integer(a) 
        state_action = torch.Tensor(np.concatenate([self.state, a]))
        with torch.no_grad():
            next_state = self.nn(state_action).numpy()
            # note that nn only predict the dynamic part of state, we need to add static part (bypass status, elapse time)
            next_state = np.concatenate([next_state, self.state[-2:]])
            cost = next_state[0] # first state var is the risk score 
            info = {} 

            # Save `terminated` and `truncated` in `info` (dict)
            info['terminated'] = False
            info['truncated'] = False 

            # set next state
            self.state = next_state

            return next_state, cost, False, False, info

    def reset(self, s_0 = None):
        if s_0 is not None:
            self.state = s_0  
            return self.state, None 
        else: 
            self.state = self.sample_state()
            return self.state, None 

    def sample_state(self):
        return np.array([np.random.uniform(min_val, max_val) for (min_val, max_val) in self.states_range])
    
    def get_total_actions(self):
        """
        Computes the total number of possible discretized actions.
        """
        total_actions = 1
        for num_discretization in self.num_discretizations:
            total_actions *= num_discretization
        return total_actions
        
    def sample_action(self):
        return np.random.randint(0, self.get_total_actions())


# def test_medical_mdp():
#     states = [(0,1)] * 10
#     actions = [(0,1)] * 6
#     num_discretizations = [2] * 6
#     kernel_path = 'medical_nn/nn_markov_model_dec.pth'
#     initial_state = np.array([0.5] * 10)
#     mdp = MedicalMDP(states=states, actions=actions, num_discretizations=num_discretizations, kernel_path=kernel_path, initial_state=initial_state)

#     # Test encoding and decoding actions
#     for action_integer in range(10):
#         action = mdp.decode_action_from_integer(action_integer)
#         action_integer_ = mdp.encode_action_to_integer(action)
#         assert action_integer == action_integer_

#     mdp.reset()
#     print(mdp.state)

#     print(mdp.sample_action())

#     # import trained kernel 
#     nn = torch.load(kernel_path, map_location=torch.device('cpu'))
#     nn.eval()
#     mdp.nn = nn 

#     # Test transition
#     state = mdp.state
#     action = mdp.sample_action()
#     next_state, reward, info = mdp.step(action)
#     print(next_state, reward, info)

# if __name__ == '__main__':
#     test_medical_mdp()

