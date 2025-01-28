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
import os

def pmd_medical_testing(env_str: str, animate: bool = False, options: dict = {}):
    """ 
    :param env_str: which gymnasium environment
    :param animate: animation mode of testing env
    """
    
    # 11 states 
    states = [
        [0, 100],
        [0, 200],
        [0, 150],
        [0, 50],
        [0, 70],
        [0, 100],
        [0, 200],
        [0, 200],
        [0, 300],
        [0, 1],
        [0, 100],
    ]
    # 19 actions
    actions = [
        [0.1, 0.75],
        [10, 1000],
        [0.1, 5],
        [25, 100],
        [1, 50],
        [0.01, 0.5],
        [0.5, 20],
        [1, 10],
        [100, 1000],
        [10, 1000],
        [0.2, 1],
        [4, 12],
        [1, 10],
        [1, 10],
        [25, 100],
        [0.01, 0.5],
        [0.1, 20],
        [0, 50],
        [100, 1000]
    ]

    num_discretizations = [2] * 8 + [1] * 11
    kernel_path = './medical_nn/nn_with_more_action_markov_model.pth'
    
    initial_state = [(up + low) / 2 for (low, up) in states]
    print(initial_state)
    initial_state[-1], initial_state[-2] = 0, 0

    mdp = MedicalMDP(states=states, actions=actions, num_discretizations=num_discretizations, kernel_path=kernel_path, initial_state=initial_state)

    # import trained kernel 
    nn = torch.load(kernel_path, map_location=torch.device('cpu'))
    print(nn)
    nn.eval()
    mdp.nn = nn 

    policy = PMDGeneralPolicy(mdp, 0.99, options=options)
    pmd_po = PMDPolicyOpt(policy)
    pmd_po.learn()
    return pmd_po


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

    # Use Streamlit session state to persist the 'po' object
    if "po" not in st.session_state:
        st.session_state.po = pmd_medical_testing(args.env_str, args.animate, options)

    evaluate_policy_user_choice_ui(st.session_state.po)


def evaluate_policy_user_choice_ui(po):
    """
    Interactive policy evaluation with action selection (Streamlit UI).
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
    history = []

    # Initialize placeholders for dynamic updates
    state_placeholder = st.empty()
    action_placeholder = st.empty()
    feedback_placeholder = st.empty()

    with torch.no_grad():
        iteration = 0  # Counter for unique keys
        po.policy.mdp.reset()  # Reset the MDP to initial state
        while True:  # Continue until user exits
            state = po.policy.mdp.state

            # Get top 3 actions and their probabilities
            top_actions, top_probs = get_top_actions(po, state, top_k=3)

            # Update placeholders for state and actions
            with state_placeholder.container():
                st.subheader("Current State:")
                print_state(state)  # Display state variables in a readable format

            with action_placeholder.container():
                st.subheader("Top 3 Recommended Actions")
                # Decode the actions from integer indices
                decoded_actions = [po.policy.mdp.decode_action_from_integer(top_actions[i]) for i in range(len(top_actions))]

                # Create a DataFrame where each decoded action is a row, and columns are action names
                action_df = pd.DataFrame(decoded_actions, columns=action_names)

                action_options = [
                    (f"Action {i+1}: {action_df.iloc[i].to_dict()}", i)
                    for i in range(len(top_actions))
    ]

                # Add exit option with value -1
                action_options = [("Exit", -1)] + action_options

                # Display radio buttons for selecting an action
                action_choice = st.radio(
                    "Select an Action:",
                    options=action_options,
                    format_func=lambda x: x[0],  # Display only the description
                    key=f"radio_{iteration}"  # Ensure a unique key for each loop iteration
                )
            iteration += 1  # Increment iteration counter for unique keys

            # Get selected action index
            selected_action_index = action_choice[1]  # Get the second element (index)

            # Check if the user chose to exit
            if selected_action_index == -1:
                feedback_placeholder.empty()  # Clear any feedback before exiting
                break

            # Get the selected action using the index
            selected_action = top_actions[selected_action_index]

            # Transition to the next state
            action_val = po.policy.mdp.decode_action_from_integer(selected_action)
            _ = po.policy.mdp.step(selected_action)

            # Log history
            history.append(create_history_entry(state, action_val))


    # Save history when done
    save_history(history)



def create_history_entry(state, action_val):
    """Create a history dataframe entry"""
    return pd.DataFrame([np.concatenate([state, action_val])], 
                       )

def save_history(history):
    """Save interaction history to Excel"""
    if history:
        pd.concat(history).to_csv("interactive_history.csv", index=False)
        print("History saved to interactive_history.csv")


def print_state(state):
    """Display state variables in readable format"""
    state_vars = [
        ('Risk Score', state[0], ''),
        ('Heart Rate', state[1], 'bpm'),
        ('PA Systolic', state[2], 'mmHg'),
        ('PEEP', state[3], 'cmH2O'),
        ('Peak Insp Pressure', state[4], 'cmH2O'),
        ('SPO2', state[5], '%'),
        ('Mean BP', state[6], 'mmHg'),
        ('Diastolic BP', state[7], 'mmHg'),
        ('Systolic BP', state[8], 'mmHg'),
    ]
    
    # Print each state variable in Streamlit
    for name, value, unit in state_vars:
        st.write(f"{name}: {value:.2f} ({unit})")



if __name__ == '__main__':
    # main()
    main_user_choice()
