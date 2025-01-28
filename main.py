from fopo import PMDTabularPolicy, PMDPolicyOpt, PMDGeneralPolicy, PDAPolicyOpt, PDATabularPolicy, PDAGeneralPolicy
from fopo import utils

import argparse
import numpy as np
import gymnasium as gym
from fopo.mdp import MountainCar, LunarLander, CartPole, GymnasiumTaxi
from fopo.mdp.medical_mdp import MedicalMDP

def main():
    parser = argparse.ArgumentParser(
                    prog='PMD Gymnasium',
                    description='Settings for solving gymnasium with PMD')

    parser.add_argument('--env_str', help="Gymnasium environment")
    parser.add_argument('--animate', action='store_true', default=False, help="Animate gymnasium during testing")
    parser.add_argument('--scale_cost', action='store_true', default=False)
    parser.add_argument('--scale_obs', action='store_true', default=False)
    parser.add_argument('--clip_cost', action='store_true', default=False)
    parser.add_argument('--clip_obs', action='store_true', default=False)

    args = parser.parse_args()

    options = {
        "cost_scaling": args.scale_cost,
        "obs_scaling": args.scale_obs,
        "cost_clip": args.clip_cost,
        "obs_clip": args.clip_obs,
    }

    pmd_gymnasium_testing(args.env_str, args.animate, options)   
    # pda_gymnasium_testing(args.env_str, args.animate)   


def pmd_medical_testing(env_str: str, animate: bool = False, options: dict = {}, model_path = None):
    """ 
    :param env_str: which gymnasium environment
    :param animate: animation mode of testing env
    """

    states = [(0,1)] * 10
    actions = [(0,1)] * 6
    num_discretizations = [2] * 6
    kernel_path = 'medical_nn/nn_markov_model_dec.pth'
    initial_state = np.array([0.5] * 10)
    discount_factor = 0.99
    mdp = MedicalMDP(discount_factor, states=states, actions=actions, num_discretizations=num_discretizations, kernel_path=kernel_path, initial_state=initial_state)

    policy = PMDGeneralPolicy(mdp, 0.995, options=options)
    pmd_po = PMDPolicyOpt(policy)
    pmd_po.learn()

    print("finished. testing policy")

    # policy_fn = lambda obs: policy.sample(obs)
    # test_policy(policy_fn, env_str, animate)


def pmd_gymnasium_testing(env_str: str, animate: bool = False, options: dict = {}):
    """ Basic testing of PMD (as of Apr 24, 2023)
    :param env_str: which gymnasium environment
    :param animate: animation mode of testing env
    """
    # import training environment and policy
    print("testing pmd")
    if env_str == 'taxi':
        mdp = gym.make("Taxi-v3")
        mdp = utils.modify_reset(mdp)
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PMDTabularPolicy(mdp, 0.99)
    elif env_str == 'mountain_car':
        mdp = gym.make("MountainCar-v0")
        mdp = utils.modify_reset(mdp)
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PMDGeneralPolicy(mdp, 0.995, options=options)
    elif env_str == 'lunar_lander':
        mdp = gym.make("LunarLander-v2")
        mdp = utils.modify_reset(mdp)
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PMDGeneralPolicy(mdp, 0.995, options=options)
    elif env_str == 'cart_pole':
        mdp = gym.make("CartPole-v1")
        mdp = utils.modify_reset(mdp)
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PMDGeneralPolicy(mdp, 0.99, options=options)
    else:
        if env_str is None:
            print("Pass in environment string with --env_str ...")
        else:
            print(f"Mode {env_str} invalid")
        exit(0)

    # pmd_po = PMDPolicyOpt(policy, load_fname="test_save.npz")
    pmd_po = PMDPolicyOpt(policy)

    # Example code
    mdp.reset()
    s = 0
    a = policy.sample(s)
    mdp.step(a)
    pmd_po.learn()

    print("finished. testing policy")

    policy_fn = lambda obs: policy.sample(obs)
    test_policy(policy_fn, env_str, animate)

def pda_gymnasium_testing(env_str: str, animate: bool = False):
    """ Basic testing of PMD (as of Apr 24, 2023)
    :param env_str: which gymnasium environment
    :param animate: animation mode of testing env
    """
    # import training environment and policy
    print("testing pda")
    if env_str == 'taxi':
        mdp = gym.make("Taxi-v3")
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PDATabularPolicy(mdp, 0.99)
    elif env_str == 'mountain_car':
        mdp = gym.make("MountainCar-v0")
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PDAGeneralPolicy(mdp, 0.995)
    elif env_str == 'lunar_lander':
        mdp = gym.make("LunarLander-v2")
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PDAGeneralPolicy(mdp, 0.995)
    elif env_str == 'cart_pole':
        mdp = gym.make("CartPole-v1")
        mdp = gym.wrappers.TransformReward(mdp, lambda r: -r)
        policy = PDAGeneralPolicy(mdp, 0.99)
    else:
        if env_str is None:
            print("Pass in environment string with --env_str ...")
        else:
            print(f"Mode {env_str} invalid")
        exit(0)

    pda_po = PDAPolicyOpt(policy)

    # Example code
    mdp.reset()
    s = 0
    a = policy.sample(s)
    mdp.step(a)
    pda_po.learn()

    print("finished. testing policy")

    policy_fn = lambda obs: policy.sample(obs)
    test_policy(policy_fn, env_str, animate)



def test_policy(policy_fn, env_str: str, animate: bool = False):
    """ Tests policy using sequential random seeds
    :param policy_fn: function that takes in state (obs) and returns action
    :param env_str: which gymnasium environment
    :param animate: animation mode of testing env
    """
    # import MDP for training and animation
    # if env_str == 'taxi':
    #     viz_mdp = GymnasiumTaxi(discount_factor=0.99, animate=animate)
    # elif env_str == 'mountain_car':
    #     viz_mdp = MountainCar(discount_factor=0.995, animate=animate)
    # elif env_str == 'lunar_lander':
    #     viz_mdp = LunarLander(discount_factor=0.995, animate=animate)
    # elif env_str == 'cart_pole':
    #     viz_mdp = CartPole(discount_factor=0.99, animate=animate)
    if env_str == 'taxi':
        # viz_mdp = gym.make("Taxi-v3", render_mode="human")
        viz_mdp = gym.make("Taxi-v3")
    elif env_str == 'mountain_car':
        # viz_mdp = gym.make("MountainCar-v0", render_mode="human")
        viz_mdp = gym.make("MountainCar-v0")
    elif env_str == 'lunar_lander':
        # viz_mdp = gym.make("LunarLander-v2", render_mode="human")
        viz_mdp = gym.make("LunarLander-v2")
    elif env_str == 'cart_pole':
        # viz_mdp = gym.make("CartPole-v1", render_mode="human")
        viz_mdp = gym.make("CartPole-v1")
    else:
        if env_str is None:
            print("Pass in environment string with --env_str ...")
        else:
            print(f"Mode {env_str} invalid")
        exit(0)
    viz_mdp = gym.wrappers.TransformReward(viz_mdp, lambda r: -r)

    print(f"Testing policy on {env_str}. LOWER score is BETTER")

    obs, _ = viz_mdp.reset(seed=0)
    seed = 1
    cum_reward = 0
    t = 0
    prev_timestep = 0

    reward_arr = np.zeros(1024)
    num_eps = 0

    for timestep in range(20000):
        action = policy_fn(obs)
        obs, rewards, terminated, truncated, info = viz_mdp.step(action)
        # cum_reward += (0.99)**t * rewards
        cum_reward += rewards
        t += 1
        if terminated:
            # print(f"Resetting at step {ct} with reward {cum_reward}")
            print(f"Epsiode reward {cum_reward}")
            print("Episode length {}".format(timestep - prev_timestep))
            reward_arr[num_eps] = cum_reward
            num_eps += 1
            if num_eps == len(reward_arr):
                reward_arr= np.append(reward_arr, np.zeros(num_eps))

            prev_timestep = timestep
            obs, _ = viz_mdp.reset(seed=seed)
            seed += 1
            cum_reward = 0
            t = 0

    # print summary statistic
    print(f"mean: {np.mean(reward_arr[:num_eps])}")
    print(f"std:  {np.std(reward_arr[:num_eps])}")


# def main():
#     mdp = MountainCar(discount_factor=0.995)
#     ### CARE: mountaincar has very sparse reward! not easy to solve ###
#     ### read: https://www.reddit.com/r/reinforcementlearning/comments/axp63j/d_state_of_the_art_deeprl_still_struggles_to/ ###
#     policy = PMDGeneralPolicy(mdp)
#     pmd_po = PMDPolicyOpt(policy)

#     # Example code
#     mdp.reset()
#     s = 0
#     a = policy.sample(s)
#     mdp.step(a)
#     pmd_po.learn()

#     print("finished")



# def main():
#     mdp = LunarLander(discount_factor=0.99)
#     # mdp = CartPole(discount_factor=0.99)
#     policy = PMDGeneralPolicy(mdp)
#     pmd_po = PMDPolicyOpt(policy)
# 
#     # Example code
#     mdp.reset()
#     s = 0
#     a = policy.sample(s)
#     mdp.step(a)
#     pmd_po.learn()
# 
#     print("finished. time to test")
# 
#     # MDP for visualization (animate=False if no animation). 
#     # vizmdp = CartPole(discount_factor=0.99, animate=True)
#     vizmdp = LunarLander(discount_factor=0.99, animate=True)
#     obs, _ = vizmdp.reset(seed=0)
#     seed = 1
#     cum_reward = 0
#     for ct in range(1000):
#         action = policy.sample(obs)
#         obs, rewards, info = vizmdp.step(action)
#         if info['terminated']:
#             print(f"Resetting at step {ct} with reward {cum_reward}")
#             obs, _ = vizmdp.reset(seed=seed)
#             seed += 1
#             cum_reward = 0
#         cum_reward += rewards

if __name__ == '__main__':
    main()
