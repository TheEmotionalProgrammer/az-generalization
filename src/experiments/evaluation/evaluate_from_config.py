import sys

sys.path.append("src/")

import os
import numpy as np
import multiprocessing
import gymnasium as gym
import wandb
import pandas as pd

from log_code.metrics import calc_metrics
from experiments.evaluation.eval_agent import eval_agent
from core.mcts import RandomRolloutMCTS
from environments.observation_embeddings import ObservationEmbedding, embedding_dict
from az.azmcts import AlphaZeroMCTS, AlphaZeroNoLoops

from az.model import (
    AlphaZeroModel,
    models_dict
)

from policies.tree_policies import tree_eval_dict
from policies.selection_policies import selection_dict_fn

from environments.register import register

import argparse
from experiments.parameters import base_parameters, env_challenges, grid_env_descriptions


from log_code.tree_visualizer import visualize_trees

def agent_from_config(hparams: dict):
    
    env = gym.make(**hparams["env_params"])

    discount_factor = hparams["discount_factor"]

    tree_evaluation_policy = tree_eval_dict(
        temperature = hparams["tree_temperature"],
    )[hparams["tree_evaluation_policy"]]

    selection_policy = selection_dict_fn(
        c = hparams["puct_c"],
    )[hparams["selection_policy"]]

    if (
        "root_selection_policy" not in hparams
        or hparams["root_selection_policy"] is None
    ):
        hparams["root_selection_policy"] = hparams["selection_policy"]


    root_selection_policy = selection_dict_fn(
        c = hparams["puct_c"],
    )[hparams["root_selection_policy"]]

    observation_embedding: ObservationEmbedding = embedding_dict[
        hparams["observation_embedding"]
    ](env.observation_space, hparams["ncols"] if "ncols" in hparams else None)

    if hparams["agent_type"] == "rollout": # Random Rollout MCTS
        if "rollout_budget" not in hparams:
            hparams["rollout_budget"] = 40
        agent = RandomRolloutMCTS(
            rollout_budget=hparams["rollout_budget"],
            root_selection_policy=root_selection_policy,
            selection_policy=selection_policy,
            discount_factor=discount_factor,
        )

    elif hparams["agent_type"] == "azmcts" or hparams["agent_type"] == "azmcts_no_loops": # AlphaZero MCTS

        filename = hparams["model_file"]

        model: AlphaZeroModel = models_dict[hparams["model_type"]].load_model(
            filename, 
            env, 
            False, 
            hparams["hidden_dim"]

        )

        model.eval()

        if "dir_epsilon" not in hparams:
            hparams["dir_epsilon"] = 0.0
            hparams["dir_alpha"] = None

        dir_epsilon = hparams["dir_epsilon"]
        dir_alpha = hparams["dir_alpha"]

        if hparams["agent_type"] == "azmcts_no_loops":
            agent = AlphaZeroNoLoops(
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
                reuse_tree=hparams["reuse_tree"],
                block_loops=hparams["block_loops"],
                loops_threshold= hparams["loops_threshold"],
            )
        elif hparams["agent_type"] == "azmcts":
            agent = AlphaZeroMCTS(
                root_selection_policy=root_selection_policy,
                selection_policy=selection_policy,
                model=model,
                dir_epsilon=dir_epsilon,
                dir_alpha=dir_alpha,
                discount_factor=discount_factor,
            )
        
    return (
        agent,
        tree_evaluation_policy,
        observation_embedding,
        hparams["planning_budget"],
    )


def eval_from_config(
    project_name="AlphaZero", entity=None, job_name=None, config=None, tags=None, eval_seed=None
):
    if tags is None:
        tags = []
    tags.append("evaluation")

    use_wandb = config["wandb_logs"]

    if use_wandb:
        # Initialize Weights & Biases
        settings = wandb.Settings(job_name=job_name)
        run = wandb.init(
            project=project_name, entity=entity, settings=settings, config=config, tags=tags
        )
        assert run is not None
        hparams = wandb.config
    else:
        hparams = config

    agent, tree_evaluation_policy, observation_embedding, planning_budget = (
        agent_from_config(hparams)
    )

    if "workers" not in hparams or hparams["workers"] is None:
        hparams["workers"] = multiprocessing.cpu_count()
    else:
        workers = hparams["workers"]

    seeds = [eval_seed] 

    test_env = gym.make(**hparams["test_env"])

    results = eval_agent(
        agent=agent,
        env=test_env,
        tree_evaluation_policy=tree_evaluation_policy,
        observation_embedding=observation_embedding,
        planning_budget=planning_budget,
        max_episode_length=hparams["max_episode_length"],
        seeds=seeds,
        temperature=hparams["eval_temp"],
        workers=workers,
        render=hparams["render"],
        return_trees=hparams["visualize_trees"] or hparams["plot_tree_densities"],
        verbose=hparams["verbose"],
    )

    if hparams["visualize_trees"]:
        results, trees = results
        trees = trees[0]
        print(f"Visualizing {len(trees)} trees...")
        visualize_trees(trees, "tree_visualizations")
            
    episode_returns, discounted_returns, time_steps, entropies = calc_metrics(
        results, agent.discount_factor, test_env.action_space.n
    )

    trajectories = []
    for i in range(results.shape[0]):
        re = []
        for j in range(results.shape[1]):
            re.append(
                observation_embedding.tensor_to_obs(results[i, j]["observations"])
            )
            if results[i, j]["terminals"] == 1:
                break
        trajectories.append(re)

    eval_res = {
        # standard logs
        "Evaluation/Mean_Returns": episode_returns.mean().item(),
        "Evaluation/Mean_Discounted_Returns": discounted_returns.mean().item(),
        "Evaluation/Mean_Timesteps": time_steps.mean().item(),

        # "Evaluation/Mean_Entropy": (th.sum(entropies, dim=-1) / time_steps).mean().item(),
        "trajectories": trajectories,
    }

    if use_wandb:
        run.log(data=eval_res)
        run.log_code(root="./src")
        # Finish the WandB run
        run.finish()
    
    else:
        print(f"Evaluation Mean Return: {eval_res['Evaluation/Mean_Returns']}")
        print(f"Evaluation Mean Discounted Return: {eval_res['Evaluation/Mean_Discounted_Returns']}")

def eval_budget_sweep(
    project_name="AlphaZeroEval",
    entity=None,
    config=None,
    budgets=None,
    num_train_seeds=None,
    num_eval_seeds=None,
    final = False,
    save = False
):
    """
    Evaluate the agent with increasing planning budgets and log the results.

    Args:
        project_name (str): WandB project name.
        entity (str): WandB entity name.
        config (dict): Base configuration for the agent.
        budgets (list): List of planning budgets to evaluate.
        num_train_seeds (int): Number of training seeds.
        num_eval_seeds (int): Number of evaluation seeds.
    """

    if config["ENV"] == "GRIDWORLD":
        if config["agent_type"] == "azmcts":
            run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_c_({config['puct_c']})_{config['map_size']}x{config['map_size']}_{config['train_config']}_{config['test_config']}"
        elif config["agent_type"] == "azmcts_no_loops":
            run_name = f"Algorithm_({config['agent_type']})_EvalPol_({config['tree_evaluation_policy']})_SelPol_({config['selection_policy']})_c_({config['puct_c']})_treuse_{config['reuse_tree']}_bloops_{config['block_loops']}_ttemp_({config['tree_temperature']})_{config['map_size']}x{config['map_size']}_{config['train_config']}_{config['test_config']}"
        else:
            raise ValueError(f"Unknown agent type: {config['agent_type']}") 
        
    if config["ENV"] == "GRIDWORLD" and config["test_env"]["deviation_type"] == "clockwise":
        run_name = "CW_" + run_name 
    
    if config["ENV"] == "GRIDWORLD" and config["test_env"]["deviation_type"] == "counter_clockwise":
        run_name = "CCW_" + run_name
    
    print(f"Run Name: {run_name}")

    if budgets is None:
        budgets = [8, 16, 32, 64, 128]  # Default budgets to sweep

    use_wandb = config["wandb_logs"]

    if use_wandb:
        run = wandb.init(
            project=project_name, 
            entity=entity, 
            name=run_name, 
            config=config, 
            tags=["budget_sweep"]
        )
        hparams = wandb.config
    else:
        hparams = config

    register()  # Register custom environments

    # Store results for plotting
    results_data = []

    for model_seed in range(num_train_seeds):
        print(f"Training Seed: {model_seed}")

        if config["ENV"] == "GRIDWORLD":
            if config["map_size"] == 8 and config["train_config"] == "NO_OBST":
                model_file = f"hyper/AZTrain_env=8x8_NO_OBST_evalpol=visit_iterations=50_budget=64_df=0.95_lr=0.001_nstepslr=2_seed={model_seed}/checkpoint.pth"
            elif config["map_size"] == 16 and config["train_config"] == "NO_OBST":
                model_file = f"hyper/AZTrain_env=16x16_NO_OBST_evalpol=visit_iterations=60_budget=128_df=0.95_lr=0.003_nstepslr=2_seed={model_seed}/checkpoint.pth"
            elif config["map_size"] == 8 and config["train_config"] == "MAZE_RL":
                model_file = f"hyper/AZTrain_env=8x8_MAZE_RL_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={model_seed}/checkpoint.pth"
            elif config["map_size"] == 8 and config["train_config"] == "MAZE_LR":
                model_file = f"hyper/AZTrain_env=8x8_MAZE_LR_evalpol=visit_iterations=100_budget=64_df=0.95_lr=0.001_nstepslr=2_c=0.5_seed={model_seed}/checkpoint.pth"
            elif config["map_size"] == 16 and config["train_config"] == "MAZE_LR":
                model_file = f"hyper/AZTrain_env=16x16_MAZE_LR_evalpol=visit_iterations=150_budget=64_df=0.95_lr=0.003_nstepslr=2_c=0.2_seed={model_seed}/checkpoint.pth"

        for budget in budgets:
            eval_results = []  # Store results across evaluation seeds for a given training seed

            for seed in range(num_eval_seeds):
                config_copy = dict(hparams)
                config_copy["model_file"] = model_file
                config_copy["planning_budget"] = budget

                #print(f"Running evaluation for planning_budget={budget}")

                agent, tree_evaluation_policy, observation_embedding, _ = agent_from_config(config_copy)
                test_env = gym.make(**config_copy["test_env"])
                seeds = [seed]

                results = eval_agent(
                    agent=agent,
                    env=test_env,
                    tree_evaluation_policy=tree_evaluation_policy,
                    observation_embedding=observation_embedding,
                    planning_budget=budget,
                    max_episode_length=config_copy["max_episode_length"],
                    seeds=seeds,
                    temperature=config_copy["eval_temp"],
                    workers=config_copy["workers"],
                    verbose=config_copy["verbose"],
                )

                episode_returns, discounted_returns, time_steps, _ = calc_metrics(
                    results, agent.discount_factor, test_env.action_space.n
                )

                eval_results.append([
                    discounted_returns.mean().item(),
                    episode_returns.mean().item(),
                    time_steps.mean().item(),

                ])

            # Compute mean across evaluation seeds for this training seed
            eval_results = np.array(eval_results)

            train_seed_mean = eval_results.mean(axis=0)  # Mean of evaluation seeds
            results_data.append([budget, model_seed] + list(train_seed_mean))
   
    if use_wandb:
        run.finish()

    # Convert results into DataFrame
    df = pd.DataFrame(results_data, columns=["Budget", "Training Seed", "Discounted Return", "Return", "Episode Length"])

    # Compute final mean and standard deviation across training seeds
    df_grouped = df.groupby("Budget").agg(["mean", "std"])

    df_grouped.columns = [f"{col[0]} {col[1]}" for col in df_grouped.columns]  # Flatten MultiIndex
    df_grouped.reset_index(inplace=True)  # Restore "Budget" as a column

    # Compute standard error across training seeds
    num_train_seeds = len(df["Training Seed"].unique())
    for metric in ["Discounted Return", "Return", "Episode Length"]:
        df_grouped[f"{metric} SE"] = df_grouped[f"{metric} std"] / np.sqrt(num_train_seeds)
    
    # Drop the "Training Seed mean" column and the "Training Seed std" column
    df_grouped.drop(columns=["Training Seed mean", "Training Seed std"], inplace=True)

    if save:
        if final:
            # Check if the folder final exists
            if not os.path.exists("final"):
                os.makedirs("final")
            # Check if the folder map_size exists
            if not os.path.exists(f"final/{config_copy['map_size']}x{config_copy['map_size']}"):
                os.makedirs(f"final/{config_copy['map_size']}x{config_copy['map_size']}")

            # Save results
            df_grouped.to_csv(f"final/{config_copy['map_size']}x{config_copy['map_size']}/{run_name}.csv", index=False)

        else:
            # If directory does not exist, create it
            if not os.path.exists(f"{config_copy['map_size']}x{config_copy['map_size']}"):
                os.makedirs(f"{config_copy['map_size']}x{config_copy['map_size']}")

            # Save results
            df_grouped.to_csv(f"{config_copy['map_size']}x{config_copy['map_size']}/{run_name}.csv", index=False)

    # Print final averages with standard errors
    print(f"Final results for {run_name}")
    for budget in budgets:
        row = df_grouped[df_grouped["Budget"] == budget]
        print(f"Planning Budget: {budget}")
        print(f"Avg Discounted Return: {row['Discounted Return mean'].values[0]:.3f} ± {row['Discounted Return SE'].values[0]:.3f}")
        print(f"Avg Return: {row['Return mean'].values[0]:.3f} ± {row['Return SE'].values[0]:.3f}")
        print(f"Avg Episode Length: {row['Episode Length mean'].values[0]:.3f} ± {row['Episode Length SE'].values[0]:.3f}")
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AlphaZero Evaluation Configuration")

    # Environment configurations
    parser.add_argument("--ENV", type=str, default="GRIDWORLD", help="Environment name")
    parser.add_argument("--map_size", type=int, default= 8, help="Map size")
    parser.add_argument("--train_config", type=str, default= "NO_OBST", help="Config desc name")
    parser.add_argument("--test_config", type=str, default= "NARROW", help="Config desc name")

    # Run configurations
    parser.add_argument("--wandb_logs", type=bool, default= False, help="Enable wandb logging")
    parser.add_argument("--workers", type=int, default= 1, help="Number of workers")

    # Standard AZ planning parameters
    parser.add_argument("--tree_evaluation_policy", type= str, default="visit", help="Tree evaluation policy")
    parser.add_argument("--selection_policy", type=str, default="UCT", help="Selection policy")
    parser.add_argument("--puct_c", type=float, default= 0, help="PUCT parameter")
    parser.add_argument("--planning_budget", type=int, default = 64, help="Planning budget")

    # Stochasticity parameters
    parser.add_argument("--eval_temp", type=float, default= 0, help="Temperature in tree evaluation softmax")
    parser.add_argument("--dir_epsilon", type=float, default= 0.0, help="Dirichlet noise parameter epsilon")
    parser.add_argument("--dir_alpha", type=float, default= None, help="Dirichlet noise parameter alpha")
    parser.add_argument("--tree_temperature", type=float, default= None, help="Temperature in tree evaluation softmax")

    # Planning algorithm
    parser.add_argument("--agent_type", type=str, default= "azmcts", help="Agent type")

    # Only for standard MCTS (no NN)
    parser.add_argument("--rollout_budget", type=int, default= 10, help="Rollout budget")

    # Test environment parameters
    parser.add_argument("--test_env_is_slippery", type=bool, default= False, help="Slippery environment")
    parser.add_argument("--test_env_hole_reward", type=int, default=0, help="Hole reward")
    parser.add_argument("--test_env_terminate_on_obst", type=bool, default= False, help="Terminate on hole")
    parser.add_argument("--deviation_type", type=str, default= "bump", help="Deviation type")

    # Model file
    parser.add_argument("--model_file", type=str, default= "", help="Model file")

    parser.add_argument( "--train_seeds", type=int, default=10, help="The number of random seeds to use for training.")
    parser.add_argument("--eval_seeds", type=int, default=1, help="The number of random seeds to use for evaluation.")

    # Rendering and logging
    parser.add_argument("--render", type=bool, default=True, help="Render the environment")
    parser.add_argument("--visualize_trees", type=bool, default=False, help="Visualize trees")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose output")

    parser.add_argument("--run_full_eval", type=bool, default= True, help="Run type")

    parser.add_argument("--hpc", type=bool, default=False, help="HPC flag")

    parser.add_argument("--value_estimate", type=str, default="nn", help="Value estimate method")

    parser.add_argument("--final", type=bool, default=False)

    parser.add_argument("--save", type=bool, default=True)

    parser.add_argument("--reuse_tree", type=bool, default=True, help="Update the estimator")
    parser.add_argument("--block_loops", type=bool, default=True, help="Block loops")
    
    parser.add_argument("--loops_threshold", type=float, default=0, help="Loop threshold")

    parser.add_argument("--plot_tree_densities", type=bool, default=False, help="Plot tree densities")

    parser.add_argument("--max_episode_length", type=int, default=100, help="Max episode length")

    parser.add_argument("--discount_factor", type=float, default=0.95, help="Discount factor")

    # Parse arguments
    args = parser.parse_args()

    single_train_seed = 2 # Only for single run
    single_eval_seed = 5 # Only for single run

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
        "hpc": args.hpc,
        "value_estimate": args.value_estimate,
        "visualize_trees": args.visualize_trees,
        "map_size": args.map_size,
        "train_config": args.train_config,
        "test_config": args.test_config,
        "tree_temperature": args.tree_temperature,
        "save": args.save,
        "reuse_tree": args.reuse_tree,
        "plot_tree_densities": args.plot_tree_densities,
        "max_episode_length": args.max_episode_length,
        "rollout_budget": args.rollout_budget,
        "block_loops": args.block_loops,
        "loops_threshold": args.loops_threshold,

    }

    run_config = {**base_parameters, **challenge, **config_modifications}

    # Execute the evaluation

    if args.run_full_eval:
        eval_budget_sweep(config=run_config, budgets= [64],  num_train_seeds=args.train_seeds, num_eval_seeds=args.eval_seeds, final = args.final, save=args.save)
    else: 
        eval_from_config(config=run_config, eval_seed=single_eval_seed)
