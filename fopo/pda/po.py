from fopo.base import BasePolicyOpt, BasePolicy
import numpy as np
from typing import Any
from fopo.pmd.utils import prox_update_Tsallis_state
from fopo.pmd.utils import random_fourier_feature, random_stump_feature, softmax
import json
import torch

"""
status at Jul 16: discrete action space only
TODO: continuous action space
"""

class PDAPolicyOpt(BasePolicyOpt):
    def __init__(self, policy: BasePolicy):
        super().__init__(policy)
        self.params = self.read_params()
        # define solve mode: tabular/linear_approx/nn
        self.solve_by_tabular, self.solve_by_linear, self.solve_by_nn = False, False, False 

        if self.params['opt']['solve_mode'] == 'tabular':
            self.setup_tabular()

        if self.params['opt']['solve_mode'] == 'linear':
            self.setup_linear()

        if self.params['opt']['solve_mode'] == 'nn':
            self.setup_nn()

        # setup initial policy
        self.setup_init_policy() 

        # print relavant params
        print("optimization params: \n {}".format(self.params['opt']))
        eval_method = self.params["eval"]['abbrv']
        print("eval method: {}, evaluation params: \n {}".format(eval_method, self.params["eval"][eval_method]))
        print("feature specs params: \n {}".format(self.params['feature_spec']))


    def setup_general_info(self):
        """
        set up related mdp info for policy learning
        make sure mdp class has the following:        
        """
        self.mdp = self.policy.mdp
        self.states = self.mdp.states 
        self.actions = self.mdp.actions
        if self.mdp.is_tabular:
            self.n_states, self.n_actions = self.mdp.n_states, self.mdp.n_actions
        elif self.mdp.is_finite_action:
            self.n_actions = self.mdp.n_actions
        self.state_dim, self.action_dim = self.mdp.state_dim, self.mdp.action_dim

    def setup_init_policy(self):
        # initial policy (required for first eval)
        if self.n_actions > 0:
            self.policy.pi = lambda s: np.ones(self.n_actions) / self.n_actions
        else:
            raise NotImplementedError()

    def setup_tabular(self):
        """
        set up for tabular learning mode
        """
        self.solve_by_tabular = True
        # self.n_states, self.n_actions = self.policy.n_states, self.policy.n_actions
        self.q = np.zeros((self.n_states, self.n_actions))
        # initial tabular representation
        self.policy_tab_repr = np.ones((self.n_states, self.n_actions), dtype=float)/self.n_actions
        self.feature_map = self.identity_feature_map
        # fed into policy class for eval
        self.policy.feature_map = self.feature_map
        self.policy.d = self.n_states * self.n_actions
        # initialze da gradient 
        self.da_grad = np.zeros((self.n_states, self.n_actions), dtype=float)

    def setup_linear(self):
        """
        set up for linear function approximation
        """
        # only consider finite action space with kl divergence
        # TO do: tsallis divergence should use vbe type estimator (still finite action)
        self.solve_by_linear = True
        self.q_params = []
        # self.n_actions = self.policy.n_actions
        # self.state_dim, self.action_dim = self.policy.state_dim, self.policy.action_dim
        # featurize 
        self.d = self.params['feature_spec']["num_features"]
        self.feature_map = self.featurize(self.params['feature_spec'])
        # fed into policy class for eval (policy and q shares the same features)
        self.policy.feature_map = self.feature_map
        self.policy.d = self.d 
        # initialze da gradient 
        self.da_grad = np.zeros(self.d)

    def setup_nn(self):
        """
        set up for neural networks
        """
        self.solve_by_nn = True
        self.qnns = []


    def read_params(self):
        """
        reading parameters for policy eval and policy optimization from json files 
        """
        params = dict()
        if self.is_tabular:
            with open("fopo/pda/params_tabular_mdp.json") as f:
                params = json.load(f)
        else: 
            with open("fopo/pda/params_nontabular_mdp.json") as f:
                params = json.load(f)
        return params 
    

    def featurize(self, feature_spec=None):
        self.feature_name = feature_spec["feature_name"]
        return self.construct_feature_map(self.feature_name)

    def construct_feature_map(self, name):
        """
        returns the feature map (a function that maps into a vector of dimension dim)
        input of the feature map: (state, action) if name is not "identity"
        output of the feature map is the continuous representation of this pair  
        """
        feature_map_dict = {
            "fourier": random_fourier_feature,
            "stump": random_stump_feature, 
            }
        if name == 'identity':
            return self.identity_feature_map
        map = feature_map_dict[name](self.state_dim + self.action_dim, self.d)
        return map 
    
    def identity_feature_map(self, state: Any, action: Any):
        """
        return the identity feature 
        """
        feature = np.zeros(self.n_states * self.n_actions)
        feature[state * self.n_actions + action] = 1.
        return feature 

    def set_policy_spec(self, theta: Any, k: int) -> None:
        """ Updates policy lookup table. See fopo/pmd/po.py's `policy_update`
        to see how policy_spec is defined.

        Set policy spec should be handled by opt methods

        Update da gradients
        """
        if self.solve_by_tabular:
            self.q = theta.reshape(self.n_states, self.n_actions)
            self.da_grad += np.sqrt(k) * self.q 
        elif self.solve_by_linear:
            self.da_grad += np.sqrt(k) * theta
        elif self.solve_by_nn:
            self.qnns.append(theta)
        return 
    
    # TODO: Rename learn() and train()
    def train(self, k: int) -> None:
        """ One iteration of a policy optimization algorithm """
        # TODO: Include logging and callbacks

        theta = self.policy.policy_evaluate(self.params['eval'])
        self.set_policy_spec(theta, k)
        self.policy_update(theta, self.params['opt'], k)


    def learn(self, display: bool = True):
        K = self.params['opt']['steps']
        perf_eval_idx = 1
        avg_value_list = []
        # initial eval
        avg_val = self.policy.performance_eval_random_state(truncated=True)
        print("Avg value at initialization: {}".format(avg_val))
        avg_value_list.append(avg_val)
        for k in range(1, K+1):
            self.train(k)
            # performance eval at 1, 2, 4, 8, ...
            avg_val = self.policy.performance_eval_random_state(k % (int(K/10)) != 0)
            if display:
                print("Avg value at iteration {}: {}".format(k, avg_val))
            avg_value_list.append(avg_val)
            # if k == perf_eval_idx or k == K:
            #     avg_val = self.policy.performance_eval()
            #     print("Avg value at iteration {}: {}".format(k, avg_val))
            #     perf_eval_idx *= 2
        # plt.plot(avg_value_list)
        # plt.xlabel("PMD iterations")
        # plt.ylabel("average cost")
        # plt.show()
                
    def policy_update(self, theta_k: Any, params: dict, k: int) -> dict:
        """ 
        policy update for pda
        """
        if self.solve_by_tabular:
            divergence = params['divergence']
            if divergence not in params['update_method'].keys():
                raise Exception("Unknown tabular policy mode {}, \n possible options: {}".format(divergence, params['update_method'].keys()))
            prox_update = getattr(self, params['update_method'][divergence])
            self.policy_tab_repr = prox_update(np.ones((self.n_states, self.n_actions))/self.n_actions, self.da_grad/k, **params[divergence])
            self.policy.pi = lambda s: self.policy_tab_repr[s, :]

        elif self.solve_by_linear:
            # generate aggregate q at state s
            da_q = lambda s: np.stack([self.feature_map(s,a) for a in self.actions]).dot(self.da_grad)
            # update policy at state s
            self.policy.pi = lambda s: self.prox_update_kl_statewise(np.ones(self.n_actions)/self.n_actions, da_q(s)/k, **params['kl'])

        elif self.solve_by_nn:
            # update policy at state s
            # self.policy.pi = lambda s: self.prox_update_nn_euclidean_finite_action(s, **params['nn'])
            self.policy.pi = lambda s: self.prox_update_kl_statewise(np.ones(self.n_actions)/self.n_actions, self.form_da_grad_nn(s)/k, **params['kl'])
        else:
            raise ValueError("Solve mode ill-defined (tabular/linear/nn)")


    def form_da_grad_nn(self, s):
        """
        construct dual averaging gradient with nn parameterized q-function, discrete action space
        """
        da_grad = torch.zeros(self.n_actions)
        for i, qnn in enumerate(self.qnns):
            da_grad = da_grad + np.sqrt(i+1) * np.array([qnn(s, a).item() for a in self.actions])
        return da_grad

    # # def prox_update_nn_euclidean_finite_action(self, s, eta: float = 0.1, n: int = 2):
    # #     """
    # #     perform policy update of dual averaging, given historical averaged Q function parameterized by torch nn
    # #     distance generating function is entropy
    # #     action space is assumed to be finite 
        
    # #     n: # steps to solve prox subproblem
    # #     """
    # #     # assert finite action MDP 
    # #     assert self.n_actions > 0
        
    # #     pi = torch.ones(self.n_actions) / self.n_actions
    # #     pi.requires_grad_()
    # #     lr = 0.1
    # #     ergodic_pi = torch.zeros(self.n_actions)

    # #     for _ in range(n):
    # #         pi.grad.zero_()

    # #         # eval subproblem 
    # #         for i, qnn in enumerate(self.qnns):
    # #             prox_loss += eta * torch.sum([qnn(s, a) * pi[a] for a in self.actions]) * torch.sqrt(i) / len(self.qnns)
    # #         prox_loss += torch.sum(pi * pi) 

    # #         prox_loss.backward()

    # #         with torch.no_grad():
    # #             pi.data = pi.data - lr * pi.grad
    # #             pi.data = self.project_onto_simplex(pi.data)

    # #         # update ergodic sum 
    # #         # TODO
                        
    # #     # might need to return ergodic iterate
    # #     return pi.detach().numpy()
    

    # # def prox_update_nn_entropy_finite_action(self, qnn, s, eta: float = 0.1, n: int = 2):
    # #     """
    # #     perform policy update of dual averaging, given historical averaged Q function parameterized by torch nn
    # #     distance generating function is entropy
    # #     action space is assumed to be finite 
        
    # #     qnn: 
    # #     """
    # #     # assert finite action MDP 
    # #     assert self.n_actions > 0
        
    # #     pi = torch.ones(self.n_actions) / self.n_actions
    # #     pi.requires_grad_()
    # #     lr = 0.1
    # #     ergodic_pi = torch.zeros(self.n_actions)

    # #     for _ in range(n):
    # #         pi.grad.zero_()

    # #         # eval subproblem 
    # #         for i, qnn in enumerate(self.qnns):
    # #             prox_loss += eta * torch.sum([qnn(s, a) * pi[a] for a in self.actions]) * torch.sqrt(i) / len(self.qnns)
    # #         prox_loss += torch.sum(pi * torch.log(pi)) 

    # #         prox_loss.backward()

    # #         with torch.no_grad():
    # #             pi.data = pi.data - lr * pi.grad
    # #             pi.data = self.project_onto_simplex(pi.data)

    # #         # update ergodic sum 
    # #         # TODO (there is no upper bound of the gradient for prox problem, also non-smooth)

    # #     # might need to return ergodic iterate
    # #     return pi.detach().numpy()


    # def project_onto_simplex(v):
    #     # Sort the elements of v in descending order
    #     u, _ = torch.sort(v, descending=True)

    #     # Compute the cumulative sum of the sorted elements
    #     cssv = torch.cumsum(u, dim=0)

    #     # Find the first index where the cumulative sum exceeds (cssv - 1) / index
    #     rho = torch.nonzero(u > (cssv - 1) / torch.arange(1, len(v) + 1).to(v.device).float())[-1]

    #     # Calculate the threshold value
    #     theta = (cssv[rho] - 1) / (rho + 1)

    #     # Apply the projection operator
    #     proj_v = torch.clamp(v - theta, min=0)

    #     return proj_v


    def prox_update_kl(self, pi: np.array, G: np.array, eta: float = 0.1, tau: float = 0) -> np.array:
        """
        solve regularized update of the form 
        \min_{p} \eta [<G(s), p> + tau h^{p}] + D^{p}_{\pi(s)}, where h denotes negative entropy, D denotes KL divergence
        this update is vectorized
        pi: policy 
        G: Q-function
        eta: stepsize 
        tau: regularization strength
        """
        logits = (np.log(pi) - eta * G) / (tau * eta + 1)
        row_wise_max = np.max(logits, axis=1)
        # stablize before pass into exp 
        p = np.exp(logits - row_wise_max[:, None])
        row_wise_sum = np.sum(p, axis=1)
        p /= row_wise_sum[:, None]
        return p 
    
    def prox_update_kl_statewise(self, pi: np.array, G: np.array, eta: float = 0.1, tau: float = 0) -> np.array:
        """
        solve regularized update of the form 
        \min_{p} \eta [<G(s), p> + \tau h^{p}] + D^{p}_{\pi(s)}, where h denotes negative entropy, D denotes KL divergence
        this update is vectorized
        pi: policy 
        G: Q-function
        eta: stepsize 
        tau: regularization strength
        """
        logits = (np.log(pi) - eta * G) / (tau * eta + 1)
        max_logit = np.max(logits)
        # stablize before pass into exp 
        p = np.exp(logits - max_logit)
        p = p / np.sum(p)
        return p 


    def prox_update_tsallis(self, pi: np.array, G: np.array, eta: float = 0.1, index: float = 0.5, tol: float = 0.001) -> np.array:
        """
        Policy update with Tsallis divergence
        """
        p = np.zeros((self.policy.n_states, self.policy.n_actions), dtype=float)
        for s in range(self.policy.n_states):
            p[s, :] = prox_update_Tsallis_state(self.policy.pi[s, :], G[s, :], eta, index, tol)
        return p 