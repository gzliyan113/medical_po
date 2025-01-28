""" May need to run (see https://github.com/DLR-RM/stable-baselines3/pull/780)
    pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
"""
import os
import sys
import gymnasium 
sys.modules["gym"] = gymnasium
gym = gymnasium

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

from fopo import GymnasiumTaxi, MountainCar, LunarLander

from main import test_policy

# model = A2C("MlpPolicy", env, verbose=1)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#    action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

def CP():
    env = make_vec_env("CartPole-v1", n_envs=1)
    # env = gym.make('MountainCar-v0')
    env._max_episode_steps = 999
    # env.tags['wrapper_config.TimeLimit.max_episode_steps'] = 500

    model = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=256,
        ent_coef=0.00429,
        learning_rate=7.77e-05,
        n_epochs=10,
        n_steps=8,
        gae_lambda=0.9,
        gamma=0.9999,
        clip_range=0.1,
        max_grad_norm =5,
        vf_coef=0.19,
        # use_sde=True, 
        policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
        verbose=1,
        # tensorboard_log=logdir
    )
    
    # For CartPole
    local_policy_path = "CartPole_5000"
    local_tuned_policy_10_1000_path = "CartPole_10_000" # Fine tuned training parameters for 10.000 iters
    local_tuned_policy_100_000_path = "CartPole_100_000" # Fine tuned training parameters for 100.000 iters
    model_path = os.path.join("saved_policies", "sb3", local_policy_path)
    
    best_model = PPO.load(model_path, env=env)
    policy_fn = lambda obs: best_model.predict(obs)[0]
    test_policy(policy_fn, 'cart_pole', animate=True)

    """
    obs, _ = viz_env.reset(seed=0)
    seed = 1
    cum_reward = 0
    ct = 0
    while True:
        action, _states = best_model.predict(obs)
        obs, rewards, term, trunc, info, = viz_env.step(action)
        if term:
            print(f"Resetting at step {ct} with reward {cum_reward}")
            obs, _ = viz_env.reset(seed=seed)
            seed += 1
            cum_reward = 0
        cum_reward += rewards
        ct += 1
        if ct >= 1000:
            break
    """

def LL():
    env = make_vec_env("LunarLander-v2", n_envs=1)
    # env = gym.make('MountainCar-v0')
    env._max_episode_steps = 999
    # env.tags['wrapper_config.TimeLimit.max_episode_steps'] = 500

    model = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=256,
        ent_coef=0.00429,
        learning_rate=7.77e-05,
        n_epochs=10,
        n_steps=8,
        gae_lambda=0.9,
        gamma=0.9999,
        clip_range=0.1,
        max_grad_norm =5,
        vf_coef=0.19,
        # use_sde=True, 
        policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
        verbose=1,
        # tensorboard_log=logdir
    )
    
    # For CartPole
    local_policy_path = "LunarLander_7000"
    local_tuned_policy_path = "LunarLander_best_model_100_000"
    model_path = os.path.join("saved_policies", "sb3", local_policy_path)

    best_model = PPO.load(model_path, env=env)
    policy_fn = lambda obs: best_model.predict(obs)[0]
    test_policy(policy_fn, 'lunar_lander', animate=True)
    
    """
    best_model = PPO.load(model_path, env=env)
    obs, _ = viz_env.reset(seed=0)
    seed = 1
    cum_reward = 0
    ct = 0
    while True:
        action, _states = best_model.predict(obs)
        obs, rewards, term, trunc, info, = viz_env.step(action)
        # negative since in PMD we do cost minimization
        cum_reward += (-rewards)
        if term:
            print(f"Resetting at step {ct} with reward {cum_reward}")
            obs, _ = viz_env.reset(seed=seed)
            seed += 1
            cum_reward = 0
        ct += 1
        if ct >= 1000:
            break
    """

if __name__ == "__main__":
    CP()
    # LL()
