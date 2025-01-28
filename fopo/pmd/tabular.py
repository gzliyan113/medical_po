from fopo.base import BasePolicy, utils
from fopo.pmd.utils import random_fourier_feature, random_stump_feature

from typing import Any, Optional
from copy import deepcopy

import numpy as np

"""
TODO: remove redundancy that's already handled by policy opt
"""

class PMDTabularPolicy(BasePolicy):
    
    def __init__(self, mdp: Any, discount_factor: float, pi_0: Optional[Any]=None):
        """ Tabular policy, i.e., lookup table for state to action
        distributions

        :params mdp: MDP environment
        :params pi_0: Initial policy. If not set, set to uniform distribution
        """
        super().__init__(mdp, discount_factor)

        self.states = utils.convert_discrete_to_array(mdp.observation_space)
        self.actions = utils.convert_discrete_to_array(mdp.action_space)
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.state_dim = np.size(self.states[0])
        self.action_dim = np.size(self.actions[0])

    def sample(self, s) -> Any:
        """ Samples by distribution from the tabular policy at state @s. """
        # first policy is uniform
        action_distribution_at_s = self.pi(s)
        action = self.rng.choice(self.actions, p=action_distribution_at_s)
        return action
        
    def create_exploratory_policy(self, cutoff: float = 0.1):
        """ Create exploratory policy for EVRTD and EVRFTD algorithms"""
        self.exploratory_pi = np.copy(self.pi)
        cutoff = cutoff / self.n_actions
        # cutoff *= 1 - self.iter / self.max_iter
        print(f'cutoff {cutoff:.3e}')
        self.explored = False
        for i in range(self.n_states):
            A_s = self.pi[i] > cutoff/2.0
            if np.sum(A_s) != self.n_actions:
                self.explored = True
            self.exploratory_pi[i, ~A_s] = cutoff
            self.exploratory_pi[i, A_s] *= 1 - np.sum(cutoff - self.exploratory_pi[i, ~A_s])
        self.exploratory_pi/= np.sum(self.exploratory_pi, axis=1)[:, None]
        # print(np.sum(self.exploratory_policy, axis=1))
        assert np.allclose(np.sum(self.exploratory_pi, axis=1), 1)

    def policy_evaluate(self, params: dict) -> Any:
        # TODO: Get rid of the first if event after independent trajectory is
        # implemented
        abbrv_name = params['abbrv']
        if abbrv_name not in params['eval_method'].keys():
            raise Exception("Unknown tabular evaluation mode {} \n Possible options: {}".format(abbrv_name, params['eval_method'].keys()))
        eval_method = getattr(self, params['eval_method'][abbrv_name])
        return eval_method(**params[abbrv_name])
    

    def _independent_trajectories(self, traj_length: int = 2,
                num_traj: int = 1):
        """
        indepndent trajectories
        traj_length: length of traj from every (s,a) pair
        num_traj: number of trajectories starting from every (s,a) pair
        """
        discount_factor = self.discount_factor
        Q = np.zeros((self.n_states, self.n_actions), dtype=float)

        for _ in range(num_traj):
            for _, state in enumerate(self.states):
                for _, action in enumerate(self.actions):
                    state, _ = self.mdp.reset(options={"s_0": state})
                    running_action = action
                    running_state, cost, terminated, truncated, info = self.mdp.step(running_action)
                    accum_cost = cost 
                    for t in range(traj_length):
                        running_action = self.sample(running_state)
                        running_state, cost, terminated, truncated, info = self.mdp.step(running_action)
                        accum_cost += (discount_factor)**t * cost
                    Q[state, action] += accum_cost

        # Average over trajectories
        Q /= num_traj
        return Q


    def _independent_trajectories_value(self, traj_length: int = 2):
        """
        indepndent trajectories
        traj_length: length of traj from every (s,a) pair
        """
        discount_factor = self.discount_factor
        sample_v = [] 

        # evaluate from a random starting state
        for i in range(100):
            state, _ = self.mdp.reset()
            accum_cost = 0 
            running_state = state
            for t in range(traj_length):
                action = self.sample(running_state)
                running_state, cost, terminated, truncated, info = self.mdp.step(action)
                accum_cost += (discount_factor)**t * cost
            sample_v.append(accum_cost)

        return np.array(sample_v)


    def performance_eval_random_state(self, truncated=True):
        """
        Eval the policy using independent trajectories to get an accurate estimation of the value function (statewise) up to target level epsilon 
        """
        # NOTE: a potentially good strategy, set alpha small here when tuning the parameters so that we have fast evaluation and a sense of whether policy 
        # is optimizing initial steps 
        alpha = 1
        if truncated:
            alpha = 0.01
            print("truncated perf eval")
        traj_length = int(alpha / (1- self.discount_factor)) + 1        
        sample_v = self._independent_trajectories_value(traj_length)
        
        # return averaged values
        return np.mean(sample_v)
    
    
    def performance_eval(self, epsilon: float = 0.5):
        """
        Eval the policy using independent trajectories to get an accurate estimation of the value function (statewise) up to target level epsilon 
        """
        traj_length = 4 * (int(1.0 / (1- self.discount_factor)) + 1)
        num_traj = int(1.0 / epsilon**2) + 1
        
        Q = self._independent_trajectories(traj_length, num_traj)
        V = np.sum(np.multiply(Q, self.pi), axis=1)
        
        # return averaged values
        return np.mean(V)

    def _td(self):
        raise NotImplementedError

    def _ctd(self, discount_or_average: bool = True, 
                explore: bool = False,
                trajectory_len: int = 30000,
                tau: int = 3,
                n_t: int = 1,
                mu: float = 0.1,
                eta: float = 1.0,
                initial_theta: Optional[np.array] = None,
                state: Optional[Any] = None, 
                discount_factor: Optional[float] = None):
        """
        General CTD function (taking TD as a special case when tau==0)
        """
        if initial_theta is None:
            initial_theta = self.rng.random(self.d)
        if explore:
            self.create_exploratory_policy()
            assert tau >= 1
    
        assert discount_or_average == True
        if not discount_factor:
            discount_factor = self.discount_factor
        self.discount_factor = discount_factor
        if state:
            state, info = self.mdp.reset(options={"s_0": state})
        else:
            state, info = self.mdp.reset()
        
        T = trajectory_len // (n_t * (tau + 1))
        t_0 = 4 / mu / eta
        theta_t = deepcopy(initial_theta)
        
        action = self.sample(state)
        next_state, cost, terminated, truncated, info = self.mdp.step(action)
        old_trajectory = [(state, action, cost)]
        state = next_state
        
        for t in range(T):
            operators, state, old_trajectory = self.minibatch_operator(state, old_trajectory, [theta_t], n_t, discount_or_average, explore, tau)
            F_t = operators[0]
            # theta_t -= 4 / mu /(t_0 + t) * F_t
            theta_t -= eta * F_t
                
        return theta_t

    def _ftd(self, discount_or_average: bool = True, 
                explore: bool = False,
                trajectory_len: int = 30000,
                tau: int = 3,
                n_t: int = 1,
                mu: float = 0.1,
                eta: float = 1.0,
                initial_theta: Optional[np.array] = None,
                state: Optional[Any] = None, 
                discount_factor: Optional[float] = None):
        """
        General FTD function
        """
        if initial_theta is None:
            initial_theta = self.rng.random(self.d)
        if explore:
            self.create_exploratory_policy()
            assert tau >= 1
    
        assert discount_or_average == True
        if not discount_factor:
            discount_factor = self.discount_factor
        self.discount_factor = discount_factor
        if state:
            state, info = self.mdp.reset(options={"s_0": state})
        else:
            state, info = self.mdp.reset()
        
        T = trajectory_len // (n_t * (tau + 1))
        t_0 = 4 / mu / eta
        theta_t = deepcopy(initial_theta)
        
        action = self.sample(state)
        next_state, cost, terminated, truncated, info = self.mdp.step(action)
        old_trajectory = [(state, action, cost)]
        state = next_state
        for t in range(T):
            operators, state, old_trajectory = self.minibatch_operator(state, old_trajectory, [theta_t], n_t, discount_or_average, explore, tau)
            F_t = operators[0]
            if t == 0:
                F_t_old = F_t
            lambda_t = 1 - 2 / (t + t_0 + 2)
            theta_t -= 4 / mu /(t_0 + t) * (F_t + lambda_t * (F_t - F_t_old))
            F_t_old = F_t
        
        return theta_t

    def _vrtd(self,  discount_or_average: bool = True, 
                     explore: bool = False,
                     trajectory_len: int = 30000,
                     tau: int = 3,
                     tau_prime: int = 3,
                     eta: float = 0.1, # stepsize parameter
                     initial_theta: Optional[np.array] = None,
                     state: Optional[Any] = None, 
                     discount_factor: Optional[float] = None):
        """
        General VRTD function, including both VRTD and EVRTD for both DMDP and AMDP
        """
        if initial_theta is None:
            initial_theta = self.rng.random(self.d)
        if explore:
            self.create_exploratory_policy()
            assert tau >= 1
            assert tau_prime >= 1
        if discount_or_average:
            if not discount_factor:
                discount_factor = self.discount_factor
            self.discount_factor = discount_factor
        K = 10
        N_k = 300
        N_k_prime = 300
        if discount_or_average:
            T = trajectory_len // K - N_k * (tau_prime + 1)
        else:
            T = trajectory_len // K - N_k * (tau_prime + 1) - N_k_prime * (tau_prime + 1)
        assert T > 1
        step_size = np.ones(T) * eta
        theta_hat_k = deepcopy(initial_theta)
        if state:
            state, info = self.mdp.reset(options={"s_0": state})
        else:
            state, info = self.mdp.reset()

        for k in range(K):
            # current best estimator
            theta_tilde = deepcopy(theta_hat_k)
            
            if not discount_or_average:
                average_cost = 0.0
                average_cost_iter = 0
                i = 1
                while average_cost_iter < N_k_prime:
                    # call the MDP environment
                    action = self.sample(state)
                    next_state, cost, terminated, truncated, info = self.mdp.step(action)
                    if i % tau_prime == 0:
                        average_cost += cost
                        average_cost_iter += 1
                    i += 1
                    old_trajectory = [(state, action, cost)]
                    state = next_state
                assert (average_cost_iter == N_k_prime)
                if N_k_prime != 0:
                    average_cost /= float(N_k_prime)
                self.average_cost = average_cost
            else:
                action = self.sample(state)
                next_state, cost, terminated, truncated, info = self.mdp.step(action)
                old_trajectory = [(state, action, cost)]
                state = next_state

            # for operator estimator centering
            average_operators, state, old_trajectory = self.minibatch_operator(state, old_trajectory, [theta_tilde], N_k, discount_or_average, explore, tau_prime)
            average_operator = average_operators[0]

            theta_k_output = np.zeros(self.d)
            step_size_sum = 0
            theta_t = deepcopy(theta_tilde)
            temp_trajectory = old_trajectory
            for t in range(T):
                if t % (tau+1) == 0 and ((t > 0) or (explore == False)):
                    if explore:
                        # action = self.sample(state, self.exploratory_pi)
                        action = self.sample(state)
                    else:
                        action = self.sample(state)
                else:
                    action = self.sample(state)
                next_state, cost, terminated, truncated, info = self.mdp.step(action)
                temp_trajectory.append((state, action, cost))

                if (t-1) % (tau+1) == 0 and ((t > 1) or (explore == False)):
                    s, a, c = temp_trajectory[-2]
                    s_prime, a_prime, _ = temp_trajectory[-1]
                    
                    if discount_or_average:
                        g_t = self.td_operator_discount(theta_t, self.discount_factor, s, a, s_prime, a_prime, c)
                        g_tilde = self.td_operator_discount(theta_tilde, self.discount_factor, s, a, s_prime, a_prime, c)
                    else:
                        g_t = self.td_operator_avg(theta_t, self.average_cost, s, a, s_prime, a_prime, c)
                        g_tilde = self.td_operator_avg(theta_tilde, self.average_cost, s, a, s_prime, a_prime, c)
                    theta_t -= eta * (g_t - g_tilde + average_operator)
                    theta_k_output += step_size[t] * theta_t
                    step_size_sum += step_size[t]
                    temp_trajectory = [temp_trajectory[-1]]
                state = next_state

            theta_hat_k = theta_k_output / step_size_sum
            temp_trajectory.clear()
        
        return theta_hat_k

    def _vrftd(self, discount_or_average: bool = True, 
                     explore: bool = False,
                     trajectory_len: int = 30000,
                     tau: int = 0,
                     tau_prime: int = 0,
                     eta: float = 0.1, # stepsize parameter
                     lambd: float = 1, # extrapolation parameter
                     initial_theta: Optional[np.array] = None,
                     state: Optional[Any] = None, 
                     discount_factor: Optional[float] = None):
        """
        General VRFTD function, including both VRFTD and EVRFTD for both DMDPs and AMDPs
        """
        if initial_theta is None:
            initial_theta = self.rng.random(self.d)
        if explore:
            self.create_exploratory_policy()
            assert tau >= 1
            assert tau_prime >= 1
        if discount_or_average:
            if not discount_factor:
                discount_factor = self.discount_factor
            self.discount_factor = discount_factor
        K = 10
        N_k = 300
        n_t = 10
        N_k_prime = 300
        if discount_or_average:
            T = (trajectory_len // K - N_k * (tau_prime + 1)) // ((tau + 1) * n_t)
        else:
            T = (trajectory_len // K - N_k * (tau_prime + 1) - N_k_prime * (tau_prime + 1)) // ((tau + 1) * n_t)
        assert T > 1
        step_size = np.ones(T) * eta
        theta_hat_k = deepcopy(initial_theta)
        if state:
            state, info = self.mdp.reset(options={"s_0": state})
        else:
            state, info = self.mdp.reset()
        for k in range(K):
            # current best estimator
            theta_tilde = deepcopy(theta_hat_k)

            if not discount_or_average:
                average_cost = 0.0
                average_cost_iter = 0
                i = 1
                while average_cost_iter < N_k_prime:
                    # call the MDP environment
                    action = self.sample(state)
                    next_state, cost, termianted, truncated, info = self.mdp.step(action)
                    if i % (tau_prime + 1) == 0:
                        average_cost += cost
                        average_cost_iter += 1
                    i += 1
                    old_trajectory = [(state, action, cost)]
                    state = next_state
                assert (average_cost_iter == N_k_prime)
                if N_k_prime != 0:
                    average_cost /= float(N_k_prime)
                self.average_cost = average_cost
            else:
                action = self.sample(state)
                next_state, cost, terminated, truncated, info = self.mdp.step(action)
                old_trajectory = [(state, action, cost)]
                state = next_state

            # for operator estimator centering
            average_operators, state, old_trajectory = self.minibatch_operator(state, old_trajectory, [theta_tilde], N_k, discount_or_average, explore, tau_prime)
            average_operator = average_operators[0]

            theta_k_output = np.zeros(self.d)
            step_size_sum = 0
            theta_t = deepcopy(theta_tilde)
            for t in range(T):
                operators, state, old_trajectory = self.minibatch_operator(state, old_trajectory, [theta_t, theta_tilde], n_t, discount_or_average, explore, tau)
                F_t = operators[0] - operators[1] + average_operator
                if t == 0:
                    F_t_old = F_t
                theta_t -= eta * (F_t + lambd * (F_t - F_t_old))
                F_t_old = F_t
                theta_k_output += step_size[t] * theta_t
                step_size_sum += step_size[t]

            theta_hat_k = theta_k_output / step_size_sum

        return theta_hat_k
    
    def _evrtd(self):
        raise NotImplementedError
    
    def td_operator_discount(self, theta: np.array, discount_factor: float, 
                             state: Any, action: Any, next_state: Any, next_action: Any, cost: float) -> np.array:
        """ TD operator with linear function approximation for discounted MDPs
        """
        sa_feat = self.feature_map(state, action)
        sa_p_feat = self.feature_map(next_state, next_action)
        return sa_feat * ((sa_feat - discount_factor * sa_p_feat).dot(theta) - cost)
        
    def td_operator_avg(self, theta: np.array, avg_cost: float, 
                        state: Any, action: Any, next_state: Any, next_action: Any, cost:float) -> np.array:
        """ TD operator with linear function approximation for average cost MDPs
        """
        sa_feat = self.feature_map(state, action)
        sa_p_feat = self.feature_map(next_state, next_action)
        return sa_feat * ((sa_feat - sa_p_feat).dot(theta) - cost + avg_cost)
    
    def minibatch_operator(self, state: Any, old_trajectory: list, thetas: list, batch_size: int = 10, discount_or_average: bool = True, explore: bool = False, tau: int = 0):
        """ This function is used to calculate the mini-batch operator of a group of theta's. It will be heavily used in VRTD/EVRTD/VRFTD
        """
        assert thetas[0].shape[0] == self.d
        assert batch_size > 0
        if explore:
            assert tau >= 1
        num_thetas = len(thetas)
        state, info = self.mdp.reset(options={"s_0": state})
        average_operators = [np.zeros(self.d) for i in range(num_thetas)]
        average_operator_iter = 0
        i = 0
        temp_trajectory = old_trajectory
        while average_operator_iter < batch_size:
            if i % (tau+1) == 0 and ((i > 0) or (explore == False)):
                if explore:
                    # action = self.sample(state, self.exploratory_pi)
                    action = self.sample(state)
                else:
                    action = self.sample(state)
            else:
                action = self.sample(state)
            next_state, cost, terminated, truncated, info = self.mdp.step(action)
            temp_trajectory.append((state, action, cost))

            # if i'th is end of consecutive (tau+1) steps, then access trajectory when i+1
            if (i-1) % (tau+1) == 0 and ((i > 1) or (explore == False)):
                s, a, c = temp_trajectory[-2]
                s_prime, a_prime, _ = temp_trajectory[-1]
                for j in range(num_thetas):
                    if discount_or_average:
                        operator = self.td_operator_discount(thetas[j], self.discount_factor,
                                                s, a, s_prime, a_prime, c)
                    else:
                        operator = self.td_operator_avg(thetas[j], self.average_cost,
                                                s, a, s_prime, a_prime, c)
                    average_operators[j] += operator
                average_operator_iter += 1
                temp_trajectory = [temp_trajectory[-1]]
            i += 1
            state = next_state

        for j in range(num_thetas):
            average_operators[j] /= float(batch_size)
        return average_operators, state, temp_trajectory
    
    
        
    
