from fopo import PMDTabularPolicy, PMDPolicyOpt, PMDGeneralPolicy, PDAPolicyOpt, PDATabularPolicy, PDAGeneralPolicy
from fopo import utils

import argparse
import numpy as np
import gymnasium as gym
from fopo.mdp import MountainCar, LunarLander, CartPole, GymnasiumTaxi
from fopo.mdp.medical_mdp import MedicalMDP
import torch 
from fopo.mdp.nn_model import SimpleResNet, SimpleLinearModel, ResidualBlock
import pandas as pd 
import streamlit as st


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

    po = pmd_medical_testing(args.env_str, args.animate, options)  
    evaluate_policy(po)


def pmd_medical_testing(env_str: str, animate: bool = False, options: dict = {}):
    """ 
    :param env_str: which gymnasium environment
    :param animate: animation mode of testing env
    """
    # 11 states 
    states = [(0,1)] * 11
    # 19 actions
    actions = [(0,1)] * 19
    num_discretizations = [2] * 8 + [1] * 11
    kernel_path = '../medical_nn/nn_with_more_action_markov_model.pth'
    
    initial_state = np.array([0.5] * 11)
    mdp = MedicalMDP(states=states, actions=actions, num_discretizations=num_discretizations, kernel_path=kernel_path, initial_state=initial_state)

    # import trained kernel 
    nn = torch.load(kernel_path, map_location=torch.device('cpu'))
    print(nn)
    nn.eval()
    mdp.nn = nn 

    policy = PMDGeneralPolicy(mdp, 0.99, options=options)
    pmd_po = PMDPolicyOpt(policy)
    # pmd_po.learn()
    return pmd_po

def evaluate_policy(po):
    """
    evalaute the learned policy starting from the initial state
    """
    state_names = [
        'Risk_Score',
        'Heart Rate - Anes',
        'PA Systolic - Anes',
        'PEEP - Anes',
        'Peak Inspiratory Pressure - An',
        'SPO2 - Anes',
        'Mean Blood Pressure',
        'Diastolic Blood Pressure',
        'Systolic Blood Pressure',
        'Bypass_Status',
        'Elapse_Time',

    ]
    action_names = [
        'infusion_Milrinone',
        'bolus_Epinephrine',
        'infusion_Nitroglycerin',
        'bolus_Sodium',
        'infusion_Insulin',
        'infusion_Epinephrine',
        'infusion_Nicardipine',
        'infusion_Vasopressin',
        'bolus_Nitroglycerin',
        'bolus_Norepinephrine',
        'bolus_Glycopyrrolate',
        'bolus_Dexamethasone',
        'bolus_Vasopressin',
        'bolus_Milrinone',
        'bolus_Diphenhydramine',
        'infusion_Phenylephrine',
        'bolus_Nicardipine',
        'infusion_Lidocaine',
        'bolus_Phenylephrine',
    ]
    state = po.policy.mdp.state 
    state_action_names = state_names + action_names
    history = []
    with torch.no_grad():
        for i in range(100):
            action = po.policy.sample(state)
            action_val = po.policy.mdp.decode_action_from_integer(action)
            # print info
            info = np.concatenate([state, action_val])
            info = pd.DataFrame([info], columns=state_action_names)
            history = history + [info]

            state_action = torch.Tensor(np.concatenate([state, action_val]))
            next_state = po.policy.mdp.nn(state_action)
            state = np.concatenate([next_state, state[-2:]])
    history = pd.concat(history)
    history.columns = state_action_names
    print(history)

    history.to_excel("history_full.xlsx", index=False)


def evaluate_policy_user_choice(po):
    """
    Interactive policy evaluation with action selection
    """
    state = po.policy.mdp.state.copy()
    history = []

    with torch.no_grad():
        while True:  # Continue until user exits
            # Get top 3 actions and their probabilities
            top_actions, top_probs = get_top_actions(po, state, top_k=3)
            
            # Display current state and top actions
            print("\nCurrent State:")
            print_state(state)
            
            print("\nTop 3 Recommended Actions:")
            for i in range(len(top_actions)):
                print(f"Action {i+1}.")
                print_action(po.policy.mdp.decode_action_from_integer(top_actions[i]))
                print("")
            
            # Get user choice
            choice = get_user_choice(len(top_actions))
            if choice == 0:
                print("Exiting...")
                break
                
            # Get selected action
            selected_action = top_actions[choice-1]
            
            # Transition to next state
            action_val = po.policy.mdp.decode_action_from_integer(selected_action)
            state_action = torch.Tensor(np.concatenate([state, action_val]))
            next_state = po.policy.mdp.nn(state_action).numpy()
            
            # Preserve bypass status and elapse time from previous state
            state = np.concatenate([next_state, state[-2:]])
            
            # Log history
            history.append(create_history_entry(state, action_val))
            
    # Save history when done
    save_history(history)

def get_top_actions(po, state, top_k=3):
    """
    Get top K actions by policy probability
    """
    # Get probability distribution over all actions
    all_actions = np.arange(po.policy.mdp.get_total_actions())
    probs = po.policy.pi(state)
    
    # Sort by descending probability
    sorted_indices = np.argsort(-probs)
    top_indices = sorted_indices[:top_k]
    
    return all_actions[top_indices], probs[top_indices]

def print_state(state):
    """Display state variables in readable format"""
    state_vars = [
        ('Risk Score', state[0], '0-1 scale'),
        ('Heart Rate', state[1], 'bpm'),
        ('PA Systolic', state[2], 'mmHg'),
        ('PEEP', state[3], 'cmH2O'),
        ('Peak Insp Pressure', state[4], 'cmH2O'),
        ('SPO2', state[5], '%'),
        ('Mean BP', state[6], 'mmHg'),
        ('Diastolic BP', state[7], 'mmHg'),
        ('Systolic BP', state[8], 'mmHg'),
        ('Bypass Status', state[9], '0=off, 1=on'),
        ('Elapsed Time', state[10], 'minutes')
    ]
    
    for name, value, unit in state_vars:
        print(f"{name}: {value:.2f} ({unit})")

def print_action(action_val):
    """Display action values in readable format"""
    action_vars = [
        'Milrinone Infusion',
        'Epinephrine Bolus',
        'Nitroglycerin Infusion',
        'Sodium Bolus',
        'Insulin Infusion',
        'Epinephrine Infusion',
        'Nicardipine Infusion',
        'Vasopressin Infusion',
        'Nitroglycerin Bolus',
        'Norepinephrine Bolus',
        'Glycopyrrolate Bolus',
        'Dexamethasone Bolus',
        'Vasopressin Bolus',
        'Milrinone Bolus',
        'Diphenhydramine Bolus',
        'Phenylephrine Infusion',
        'Nicardipine Bolus',
        'Lidocaine Infusion',
        'Phenylephrine Bolus',
    ]
    
    for name, value in zip(action_vars, action_val):
        print(f" - {name}: {value:.4f}")

def get_user_choice(max_options):
    """Get validated user input"""
    while True:
        try:
            choice = int(input(f"\nChoose action (1-{max_options}, 0 to exit): "))
            if 0 <= choice <= max_options:
                return choice
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def create_history_entry(state, action_val):
    """Create a history dataframe entry"""
    return pd.DataFrame([np.concatenate([state, action_val])], 
                       )

def save_history(history):
    """Save interaction history to Excel"""
    if history:
        pd.concat(history).to_excel("interactive_history.xlsx", index=False)
        print("History saved to interactive_history.xlsx")


def main_user_choice():
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

    po = pmd_medical_testing(args.env_str, args.animate, options)  
    evaluate_policy_user_choice(po)




if __name__ == '__main__':
    # main()
    main_user_choice()
