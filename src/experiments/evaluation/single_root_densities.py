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
            agent, _, _, _, planning_budget = (
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
    map_size = 8

    TRAIN_CONFIG = "MAZE_LR" # NO_OBST, MAZE_RL, MAZE_LR

    TEST_CONFIG = "MAZE_RL" # NO_OBSTS, OBST, MAZE_RL, MAZE_LR

    parser.add_argument("--ENV", type=str, default="GRIDWORLD", help="Environment name")

    parser.add_argument("--map_size", type=int, default= map_size, help="Map size")
    parser.add_argument("--test_config", type=str, default= TEST_CONFIG, help="Config desc name")
    parser.add_argument("--train_config", type=str, default= TRAIN_CONFIG, help="Config desc name")

    # Run configurations
    parser.add_argument("--wandb_logs", type=bool, default= False, help="Enable wandb logging")
    parser.add_argument("--workers", type=int, default= 1, help="Number of workers")
    parser.add_argument("--runs", type=int, default= 1, help="Number of runs")

    # Basic search parameters
    parser.add_argument("--tree_evaluation_policy", type= str, default="visit", help="Tree evaluation policy")
    parser.add_argument("--selection_policy", type=str, default="UCT", help="Selection policy")
    parser.add_argument("--puct_c", type=float, default= 0, help="PUCT parameter")

    # Only relevant for single run evaluation
    parser.add_argument("--planning_budget", type=int, default = 128, help="Planning budget")
    # Only for MCTS
    parser.add_argument("--rollout_budget", type=int, default= 100, help="Rollout budget")

    # Search algorithm
    parser.add_argument("--agent_type", type=str, default= "azmcts_no_loops", help="Agent type")

    # Stochasticity parameters
    parser.add_argument("--eval_temp", type=float, default= 0, help="Temperature in tree evaluation softmax")
    parser.add_argument("--dir_epsilon", type=float, default= 0.0, help="Dirichlet noise parameter epsilon")
    parser.add_argument("--dir_alpha", type=float, default= None, help="Dirichlet noise parameter alpha")

    parser.add_argument("--tree_temperature", type=float, default= None, help="Temperature in tree evaluation softmax")

    # AZDetection detection parameters
    parser.add_argument("--threshold", type=float, default= 0.05, help="Detection threshold")
    parser.add_argument("--unroll_budget", type=int, default= 4, help="Unroll budget")

    # AZDetection replanning parameters
    parser.add_argument("--value_search", type=bool, default=True, help="Enable value search")
    parser.add_argument("--predictor", type=str, default="current_value", help="Predictor to use for detection")
    parser.add_argument("--update_estimator", type=bool, default=True, help="Update the estimator")

    # Test environment
    parser.add_argument("--test_env_is_slippery", type=bool, default= False, help="Slippery environment")
    parser.add_argument("--test_env_hole_reward", type=int, default=0, help="Hole reward")
    parser.add_argument("--test_env_terminate_on_obst", type=bool, default= False, help="Terminate on hole")
    parser.add_argument("--deviation_type", type=str, default= "bump", help="Deviation type")

    parser.add_argument("--ll_test_config", type=str, default= "TWO_LATERAL_PENTAGONS", help="LunarLander test config")

    # Model file
    parser.add_argument("--model_file", type=str, default= "", help="Model file")

    parser.add_argument( "--train_seeds", type=int, default=10, help="The number of random seeds to use for training.")
    parser.add_argument("--eval_seeds", type=int, default=1, help="The number of random seeds to use for evaluation.")

    # Rendering
    parser.add_argument("--render", type=bool, default=False, help="Render the environment")
    parser.add_argument("--visualize_trees", type=bool, default=True, help="Visualize trees")

    parser.add_argument("--hpc", type=bool, default=False, help="HPC flag")

    parser.add_argument("--value_estimate", type=str, default="nn", help="Value estimate method")

    parser.add_argument("--final", type=bool, default=False)

    parser.add_argument("--save", type=bool, default=False)

    parser.add_argument("--reuse_tree", type=bool, default=True, help="Update the estimator")
    parser.add_argument("--block_loops", type=bool, default=False, help="Block loops")

    parser.add_argument("--plot_tree_densities", type=bool, default=False, help="Plot tree densities")

    parser.add_argument("--max_episode_length", type=int, default=100, help="Max episode length")

    parser.add_argument("--discount_factor", type=float, default=0.95, help="Discount factor")

    parser.add_argument("--subopt_threshold", type=float, default=0.1, help="Suboptimality threshold")
    parser.add_argument("--loops_threshold", type=float, default=0, help="Loop threshold")

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
        "hole_reward": args.test_env_hole_reward,
        "terminate_on_obst": args.test_env_terminate_on_obst,
        "deviation_type": args.deviation_type,
    }  

    map_name = args.test_env_desc

    # Construct the config
    config_modifications = {
        "wandb_logs": args.wandb_logs,
        "workers": args.workers,
        "runs": args.runs,
        "tree_evaluation_policy": args.tree_evaluation_policy,
        "selection_policy": args.selection_policy,
        "planning_budget": args.planning_budget,
        "discount_factor": args.discount_factor,
        "puct_c": args.puct_c,
        "agent_type": args.agent_type,
        "eval_temp": args.eval_temp,
        "dir_epsilon": args.dir_epsilon,
        "dir_alpha": args.dir_alpha,
        "threshold": args.threshold,
        "unroll_budget": args.unroll_budget,
        "value_search": args.value_search,
        "predictor": args.predictor,
        "map_name": map_name,
        "test_env": test_env_dict,
        "observation_embedding": observation_embedding,
        "model_file": args.model_file,
        "render": args.render,
        "hpc": args.hpc,
        "value_estimate": args.value_estimate,
        "visualize_trees": args.visualize_trees,
        "map_size": args.map_size,
        "update_estimator": args.update_estimator,
        "train_config": args.train_config,
        "test_config": args.test_config,
        "tree_temperature": args.tree_temperature,
        "save": args.save,
        "reuse_tree": args.reuse_tree,
        "plot_tree_densities": args.plot_tree_densities,
        "max_episode_length": args.max_episode_length,
        "rollout_budget": args.rollout_budget,
        "subopt_threshold": args.subopt_threshold,
        "block_loops": args.block_loops,
        "loops_threshold": args.loops_threshold,
    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    ROOT_STATE = (0,3)

    eval_seeds = list(range(10))
    train_seeds = list(range(10))
    create_average_density_visualizations(run_config, ROOT_STATE, train_seeds, eval_seeds)





