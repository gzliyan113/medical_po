from fopo import PMDTabularPolicy, PMDPolicyOpt, PMDGeneralPolicy
from fopo import GymnasiumTaxi, MountainCar, LunarLander, CartPole

import optuna

import numpy as np
import sys
import logging
import itertools
import json

import gymnasium as gym

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


def lunar_lander_fopo_objective(trial):
    """ Objective for Optuna tuning. Start with VRFTD """

    tau = int(trial.suggest_float("tau", 1e-1, 1e1, log=True))
    tau_prime = int(trial.suggest_float("tau_prime", 1e-1, 1e1, log=True))

    eta_actor  = trial.suggest_float("eta_actor", 1e-2, 1e1, log=True)
    eta_critic = trial.suggest_float("eta_critic", 1e-2, 1e1, log=True)

    K = trial.suggest_int("K", 1e1, 1e2, log=True)
    N_k = trial.suggest_int("N_k", 1e2, 1e3, log=True)
    n_t = trial.suggest_int("n_t", 1e0, 5e1, log=True)

    with open("fopo/pmd/params_nontabular2.json") as f:
        params = json.load(f)

    params['eval']['vrftd'] = {
        "discount_or_average": True, 
        "explore": False,
        "trajectory_len": 40000,
        "tau": tau,
        "tau_prime": tau_prime,
        "eta": eta_critic,
        "lambd": 1.0,
        "K": K,
        "N_k": N_k,
        "n_t": n_t
    }
    ntrials = 11
    eval_iter = 5
    params['opt']['eta'] = eta_actor
    params['opt']['steps'] = ntrials

    mdp = LunarLander(discount_factor=0.99)
    policy = PMDGeneralPolicy(mdp)
    pmd_po = PMDPolicyOpt(policy, params)
    avg_rwd = pmd_po.learn()

    # return last
    last_reward = avg_rwd[-1]
    return last_reward

# TODO: Move this
class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, env, scaler, featurizer):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        self.scaler = scaler
        self.featurizer = featurizer

        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    reward_arr = np.zeros(num_episodes)

    for i_episode in range(num_episodes):
        
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # Reset the environment and pick the first action
        state = env.reset()[0]
        
        # Only used for SARSA, not Q-Learning
        next_action = None
        
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            # Take a step
            next_state, reward, done, trunc, _ = env.step(action)
    
            # Update statistics
            reward_arr[i_episode] -= reward
            
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            
            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            # td_target = reward + discount_factor * q_values_next[next_action]
            
            # Update the function approximator using our target
            estimator.update(state, action, td_target)
                
            if done:
                break
            if trunc:
                state = env.reset()[0]
                
            state = next_state
    
    # return the last one
    return reward_arr[-1]

def lunar_lander_qlearn_objective(trial):
    num_episodes = 100

    discount_factor = trial.suggest_float("discf", 0.1, 1.0, log=True)
    epsilon         = trial.suggest_float("eps", 1e-4, 1e-1, log=True)
    epsilon_decay   = trial.suggest_float("eps_decay", 0.1, 1.0, log=True)

    env = gym.envs.make("LunarLander-v2")
    
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=20.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=10.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf5", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf6", RBFSampler(gamma=0.5, n_components=100)),
            ("rbf7", RBFSampler(gamma=0.1, n_components=100))
            ])
    featurizer.fit(scaler.transform(observation_examples))

    estimator = Estimator(env, scaler, featurizer)
    last_rwd = q_learning(env, estimator, num_episodes, discount_factor, epsilon, epsilon_decay)

    return last_rwd

def tune():
    study = optuna.create_study()
    print(f"Sampler is {study.sampler.__class__.__name__}")
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study.optimize(lunar_lander_fopo_objective, n_trials=20)
    # study.optimize(lunar_lander_qlearn_objective, n_trials=20)

if __name__ == "__main__":
    tune()
