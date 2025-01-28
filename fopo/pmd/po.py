from fopo.base import BasePolicyOpt, BasePolicy
import numpy as np
from fopo.pmd.utils import bisection_search
from fopo.pmd.utils import prox_update_Tsallis_state
from fopo.pmd.utils import random_fourier_feature, random_stump_feature, softmax
import json

from typing import Any
import time

"""
structure changes:
opt class handles policy representation (creates a blaxbox for policy) and update its representation
policy class handles evaluation (use the policy representation/blackbox maintained by opt class)
consequently, policy class is agnostic to tabular/non-tabular mdps 
and we can even solve tabular mdp by nontabular method by specifying this to the opt class
"""

class PMDPolicyOpt(BasePolicyOpt):
    
    def __init__(self, policy: BasePolicy, params: dict=None, **kwargs):
        super().__init__(policy)
        self.params = self.read_params() if params is None else params
        # define solve mode: tabular/linear_approx/nn
        self.solve_by_tabular, self.solve_by_linear, self.solve_by_nn = False, False, False 
        self.states = self.policy.states 
        self.actions = self.policy.actions

        save_fname = kwargs.get("save_fname", None) 
        load_fname = kwargs.get("load_fname", None) 

        if self.params['opt']['solve_mode'] == 'tabular':
            self.solve_by_tabular = True
            self.n_states, self.n_actions = self.policy.n_states, self.policy.n_actions
            self.q = np.zeros((self.n_states, self.n_actions))
            # initial tabular representation
            self.policy_tab_repr = np.ones((self.n_states, self.n_actions), dtype=float)/self.n_actions
            self.feature_map = self.identity_feature_map
            # fed into policy class for eval
            self.policy.feature_map = self.feature_map
            self.policy.d = self.n_states * self.n_actions

        if self.params['opt']['solve_mode'] == 'linear':
            # only consider finite action space with kl divergence
            # TO do: tsallis divergence should use vbe type estimator (still finite action)
            self.solve_by_linear = True
            self.q_params = []
            self.n_actions = self.policy.n_actions
            self.state_dim, self.action_dim = self.policy.state_dim, self.policy.action_dim
            # featurize 
            self.d = self.params['feature_spec']["num_features"]
            self.feature_map = self.featurize(
                self.params['feature_spec'], 
                load_fname, 
                save_fname
            )
            # fed into policy class for eval (policy and q shares the same features)
            self.policy.feature_map = self.feature_map
            self.policy.d = self.d 

        if self.params['opt']['solve_mode'] == 'nn':
            self.solve_by_nn = True

        # initial policy (required for first eval)
        self.policy.pi = lambda s: np.ones(self.n_actions) / self.n_actions

        # print relavant params
        print("optimization params: \n {}".format(self.params['opt']))
        eval_method = self.params["eval"]['abbrv']
        print("eval method: {}, evaluation params: \n {}".format(eval_method, self.params["eval"][eval_method]))
        print("feature specs params: \n {}".format(self.params['feature_spec']))

    def featurize(self, feature_spec=None, load_fname=None, save_fname=None):

        if load_fname is not None:
            map = self.load_feature_map(load_fname)
        else:
            self.feature_name = feature_spec["feature_name"]
            map, features = self.construct_feature_map(self.feature_name)

            if save_fname is not None:
                # save type, then features
                feature_id = 0 if self.feature_name == "stump" else 1
                np.savez(save_fname, feature_id, *features)
                print(f"Saved file to {save_fname}")

        return map

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
        map, features = feature_map_dict[name](self.state_dim + self.action_dim, self.d)
        return map, features

    def load_feature_map(self, load_fname):
        """ Load npz file """

        npzfile = np.load(load_fname)
        feature_name = "stump" if npzfile[npzfile.files[0]] == 0 else "fourier"

        if feature_name == "stump":
            # stump
            thresholds = npzfile[npzfile.files[1]]
            dimensions = npzfile[npzfile.files[2]]
            map, _ = random_stump_feature(-1, -1, thresholds=thresholds, dimensions=dimensions)
        else:
            W = npzfile[npzfile.files[1]]
            b = npzfile[npzfile.files[2]]
            map, _ = random_fourier_feature(-1, -1, W=W, b=b)

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
        """
        if self.solve_by_tabular:
            self.q = theta.reshape(self.n_states, self.n_actions)
        elif self.solve_by_linear:
            self.q_params.append(theta)
        return 
    
    # TODO: Rename learn() and train()
    def train(self, k: int) -> None:
        """ One iteration of a policy optimization algorithm """
        # TODO: Include logging and callbacks

        theta = self.policy.policy_evaluate(self.params['eval'])
        self.set_policy_spec(theta, k)
        self.policy_update(theta, self.params['opt'], k)

    def read_params(self):
        """
        reading parameters for policy eval and policy optimization from json files 

        # TODO: Handle case when no given parameters
        """
        params = dict()
        if self.is_tabular:
            with open("fopo/pmd/params_tabular_mdp.json") as f:
                params = json.load(f)
        else: 
            with open("fopo/pmd/params_nontabular_mdp.json") as f:
            # with open("fopo/pmd/params_nontabular2.json") as f:
                params = json.load(f)
        return params 

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
            avg_val = self.policy.performance_eval_random_state(truncated=True)
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
        return avg_value_list
                
    def policy_update(self, theta_k: Any, params: dict, k: int) -> dict:
        """ Policy update for policy mirror descent. In the tabular case, it is
        \[ 
            pi_{k+1}(s) = argmin_a \{ \phi^{\pi_k}(s,a) + h^a(s) + (1/\eta_k) D(pi_k(s), a) \},
        \]
        where $\phi^{\pi_k}(s,a)$ is the general state-action value function
        (e.g., Q- or advantage function) and $h^a$ is a convex regularization
        term.

        In the non-tabular case with function approximation parameter
        $\theta_k$, we get
        \[
            pi_{k+1}(s) = argmin_a \{ \tilde{L}(s,a; \theta_k) + h^a(s) + (1/\eta_k) w(a) \}. 
        \]
        To sample, we need to save $\theta_k$ and the step size $\eta_k$.
        """
        if self.solve_by_tabular:
            divergence = params['divergence']
            if divergence not in params['update_method'].keys():
                raise Exception("Unknown tabular policy mode {}, \n possible options: {}".format(divergence, params['update_method'].keys()))
            prox_update = getattr(self, params['update_method'][divergence])
            self.policy_tab_repr = prox_update(self.policy_tab_repr, self.q, **params[divergence])
            self.policy.pi = lambda s: self.policy_tab_repr[s, :]

        elif self.solve_by_linear:
            aggregate_q_params = sum(self.q_params)
            # generate aggregate q at state s
            aggregate_q = lambda s: np.stack([self.feature_map(s,a) for a in self.actions]).dot(aggregate_q_params)
            # update policy at state s
            self.policy.pi = lambda s: self.prox_update_kl_statewise(np.ones(self.n_actions)/self.n_actions, aggregate_q(s), **params['kl'])

        elif self.solve_by_nn:
            raise NotImplementedError
        
        else:
            raise NotImplementedError


    def prox_update_kl(self, pi: np.array, G: np.array, eta: float = 0.1, tau: float = 0) -> np.array:
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
        start_time = time.time()
        logits = (np.log(pi) - eta * G) / (tau * eta + 1)
        max_logit = np.max(logits)
        # stablize before pass into exp 
        p = np.exp(logits - max_logit)
        p = p / np.sum(p)
        # print("statewise update time: ", time.time() - start_time)
        return p 


    def prox_update_tsallis(self, pi: np.array, G: np.array, eta: float = 0.1, index: float = 0.5, tol: float = 0.001) -> np.array:
        """
        Policy update with Tsallis divergence
        """
        p = np.zeros((self.policy.n_states, self.policy.n_actions), dtype=float)
        for s in range(self.policy.n_states):
            p[s, :] = prox_update_Tsallis_state(self.policy.pi[s, :], G[s, :], eta, index, tol)
        return p 
