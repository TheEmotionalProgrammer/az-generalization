import sys

sys.path.append("src/")

import numpy as np
import torch as th
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
from experiments.parameters import base_parameters, env_challenges, grid_env_descriptions

from log_code.plot_state_densities import (
    plot_density, calculate_density
)

from experiments.evaluation.evaluate_from_config import agent_from_config

from experiments.evaluation.evaluate_from_config import register

def invert_grid_coordinate(coords, map_size):
    return coords[0] * map_size + coords[1] if coords is not None else None

def create_average_density_visualizations(run_config, root_state, train_seeds, eval_seeds):
    """
    Create density visualizations averaged across multiple training and evaluation seeds.

    Args:
        run_config (dict): Configuration for the run.
        root_state (tuple): Root state coordinates.
        train_seeds (list): List of training seeds.
        eval_seeds (list): List of evaluation seeds.
    """
    hparams = run_config
    map_size = hparams["map_size"]

    # Initialize accumulators for averaging
    accumulated_visits = None

    register()

    for train_seed in train_seeds:
        # Set model_file based on map_size and train_config
        if hparams["map_size"] == 8 and hparams["train_config"] == "NO_OBST":
            hparams["model_file"] = f"hyper/AZTrain_env=GridWorldNoObst8x8-v1_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 16 and hparams["train_config"] == "NO_OBST":
            hparams["model_file"] = f"hyper/AZTrain_env=GridWorldNoObst16x16-v1_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 8 and hparams["train_config"] == "MAZE_RL":
            hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 8 and hparams["train_config"] == "MAZE_LR":
            hparams["model_file"] = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={train_seed}/checkpoint.pth"
        elif hparams["map_size"] == 16 and hparams["train_config"] == "MAZE_LR":
            hparams["model_file"] = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={train_seed}/checkpoint.pth"

        for eval_seed in eval_seeds:
            agent, _, _, planning_budget = (
                agent_from_config(hparams)
            )

            test_env = gym.make(**hparams["test_env"])

            if eval_seed is not None:
                th.manual_seed(eval_seed)
                np.random.seed(eval_seed)

            test_env.reset(seed=eval_seed)

            test_env.unwrapped.s = invert_grid_coordinate(root_state, hparams["map_size"])
            observation = test_env.unwrapped.s

            tree = agent.search(test_env, planning_budget, observation, 0)

            test_desc = hparams["test_env"]["desc"]
            obst_coords = []
            for i in range(len(test_desc)):
                for j in range(len(test_desc[0])):
                    if test_desc[i][j] == "H":
                        obst_coords.append((i, j))

            visits  = calculate_density(tree, map_size, map_size)

            if accumulated_visits is None:
                accumulated_visits = visits
            else:
                accumulated_visits += visits

    visits_cmap = sns.diverging_palette(10, 120, as_cmap=True, center="light")
    ax = plot_density(accumulated_visits/100, tree.observation, obst_coords, map_size, map_size, cmap=visits_cmap)
    #ax.set_title(f"Averaged Visits Across Training and Evaluation Seeds")
    plt.show()
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AlphaZero Evaluation Configuration")

    # Environment configurations
    parser.add_argument("--ENV", type=str, default="GRIDWORLD", help="Environment name")
    parser.add_argument("--map_size", type=int, default= 8, help="Map size")
    parser.add_argument("--train_config", type=str, default= "MAZE_LR", help="Config desc name")
    parser.add_argument("--test_config", type=str, default= "MAZE_RL", help="Config desc name")
    parser.add_argument("--max_episode_length", type=int, default=100, help="Max episode length")

    # Run configurations
    parser.add_argument("--wandb_logs", type=bool, default= False, help="Enable wandb logging")
    parser.add_argument("--workers", type=int, default= 1, help="Number of workers")

    # Planning algorithm
    parser.add_argument("--agent_type", type=str, default= "azmcts_no_loops", help="Agent type")

    # Standard AZ planning parameters
    parser.add_argument("--tree_evaluation_policy", type= str, default="visit", help="Tree evaluation policy")
    parser.add_argument("--selection_policy", type=str, default="UCT", help="Selection policy")
    parser.add_argument("--puct_c", type=float, default= 0, help="PUCT parameter")
    parser.add_argument("--planning_budget", type=int, default = 128, help="Planning budget")
    parser.add_argument("--discount_factor", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--value_estimate", type=str, default="nn", help="Value estimate method")

    # Stochasticity parameters
    parser.add_argument("--eval_temp", type=float, default= 0, help="Temperature in tree evaluation softmax")
    parser.add_argument("--dir_epsilon", type=float, default= 0.0, help="Dirichlet noise parameter epsilon")
    parser.add_argument("--dir_alpha", type=float, default= None, help="Dirichlet noise parameter alpha")
    parser.add_argument("--tree_temperature", type=float, default= None, help="Temperature in tree evaluation softmax")

    # Only for standard MCTS (no NN)
    parser.add_argument("--rollout_budget", type=int, default= 10, help="Rollout budget")

    # Test environment parameters
    parser.add_argument("--test_env_is_slippery", type=bool, default= False, help="Slippery environment")
    parser.add_argument("--test_env_obst_reward", type=int, default=0, help="Hole reward")
    parser.add_argument("--test_env_terminate_on_obst", type=bool, default= False, help="Terminate on hole")
    parser.add_argument("--deviation_type", type=str, default= "bump", help="Deviation type")

    # Model and seeding
    parser.add_argument("--model_file", type=str, default= "", help="Model file")
    parser.add_argument( "--train_seeds", type=int, default=10, help="The number of random seeds to use for training.")
    parser.add_argument("--eval_seeds", type=int, default=10, help="The number of random seeds to use for evaluation.")
    parser.add_argument("--run_full_eval", type=bool, default= True, help="Run type")

    # Rendering and logging
    parser.add_argument("--render", type=bool, default=True, help="Render the environment")
    parser.add_argument("--visualize_trees", type=bool, default=False, help="Visualize trees")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose output")

    parser.add_argument("--save", type=bool, default=True)

    # Additional parameters for NoLoopsMCTS
    parser.add_argument("--reuse_tree", type=bool, default=True, help="Update the estimator")
    parser.add_argument("--block_loops", type=bool, default=True, help="Block loops")

    # Parse arguments
    args = parser.parse_args()

    single_train_seed = 1 # Only for single run
    single_eval_seed = 0 # Only for single run

    ENV = args.ENV

    args.test_env_id = f"GridWorldNoObst{args.map_size}x{args.map_size}-v1"
    args.test_env_desc = f"{args.map_size}x{args.map_size}_{args.test_config}"

    if args.map_size == 8 and args.train_config == "NO_OBST":
        args.model_file = f"hyper/AZTrain_env=8x8_NO_OBST_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 16 and args.train_config == "NO_OBST":
        args.model_file = f"hyper/AZTrain_env=16x16_NO_OBST_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 8 and args.train_config == "MAZE_RL":
        args.model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 8 and args.train_config == "MAZE_LR":
        args.model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={single_train_seed}/checkpoint.pth"
    elif args.map_size == 16 and args.train_config == "MAZE_LR":
        args.model_file = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={single_train_seed}/checkpoint.pth"

    challenge = env_challenges[f"GridWorldNoObst{args.map_size}x{args.map_size}-v1"]  # Training environment

    observation_embedding = "coordinate"

    test_env_dict = {
        "id": args.test_env_id,
        "desc": grid_env_descriptions[args.test_env_desc],
        "is_slippery": args.test_env_is_slippery,
        "hole_reward": args.test_env_obst_reward,
        "terminate_on_obst": args.test_env_terminate_on_obst,
        "deviation_type": args.deviation_type,
    }  

    map_name = args.test_env_desc

    # Construct the config
    config_modifications = {
        "ENV": args.ENV,
        "wandb_logs": args.wandb_logs,
        "workers": args.workers,
        "tree_evaluation_policy": args.tree_evaluation_policy,
        "selection_policy": args.selection_policy,
        "planning_budget": args.planning_budget,
        "discount_factor": args.discount_factor,
        "puct_c": args.puct_c,
        "agent_type": args.agent_type,
        "eval_temp": args.eval_temp,
        "dir_epsilon": args.dir_epsilon,
        "dir_alpha": args.dir_alpha,
        "map_name": map_name,
        "test_env": test_env_dict,
        "observation_embedding": observation_embedding,
        "model_file": args.model_file,
        "render": args.render,
        "verbose": args.verbose,
        "value_estimate": args.value_estimate,
        "visualize_trees": args.visualize_trees,
        "map_size": args.map_size,
        "train_config": args.train_config,
        "test_config": args.test_config,
        "tree_temperature": args.tree_temperature,
        "save": args.save,
        "reuse_tree": args.reuse_tree,
        "max_episode_length": args.max_episode_length,
        "rollout_budget": args.rollout_budget,
        "block_loops": args.block_loops,

    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    ROOT_STATE = (0,5)

    eval_seeds = list(range(10))
    train_seeds = list(range(10))
    create_average_density_visualizations(run_config, ROOT_STATE, train_seeds, eval_seeds)





