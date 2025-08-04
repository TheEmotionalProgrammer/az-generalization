import multiprocessing
import os
import subprocess
from tabnanny import verbose

from tensordict import TensorDict
import torch as th
import gymnasium as gym
import numpy as np
from core.mcts import MCTS, RandomRolloutMCTS, NoLoopsMCTS
from az.azmcts import AlphaZeroMCTS
from environments.observation_embeddings import ObservationEmbedding
from policies.policies import PolicyDistribution, custom_softmax
from core.node import Node
from core.utils import copy_environment, observations_equal, actions_dict, print_obs
import matplotlib.pyplot as plt


import copy

from log_code.gen_renderings import save_gif_imageio

from policies.utility_functions import get_children_visits


def collect_trajectories(tasks, workers=1):
    subprocess.run(["pwd"]) # Just running pwd somehow fixes a multiprocessing issue on HPCs.
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            # Run the tasks using map
            results = pool.map(run_episode_process, tasks)
    else:
        results = [run_episode_process(task) for task in tasks] 

    # check if the results are tuples, if so, unpack them
    if all(isinstance(result, tuple) for result in results):
        trajectories, trees = zip(*results)
        res_tensor = th.stack(trajectories)
        return res_tensor, trees
    else:
        res_tensor =  th.stack(results)
        return res_tensor

def run_episode_process(args):

    """Wrapper function for multiprocessing that unpacks arguments and runs a single episode with the specified algorithm."""

    agent = args[0]
    
    if isinstance(agent, NoLoopsMCTS):
        return run_episode_no_loop(*args)
    
    elif isinstance(agent, AlphaZeroMCTS) or isinstance(agent, RandomRolloutMCTS):
        return run_episode_azmcts(*args)
    
    else:
        raise NotImplementedError("Agent type not implemented.")
    
    
@th.no_grad()
def run_episode_azmcts(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000,
    max_steps=1000,
    seed=None,
    temperature=None,
    render=False,
    return_trees=False,
    verbose=False,
):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, _ = env.reset(seed=seed)

    if verbose:
        print(f"Env: obs = {print_obs(env, observation)}")

    if render:
        vis_env = copy_environment(env)  # Use the utility function
        vis_env.unwrapped.render_mode = "rgb_array"
        frames = [vis_env.render()]

    observation_tensor: th.Tensor = observation_embedding.obs_to_tensor(observation, dtype=th.float32)

    trajectory = TensorDict(
        source={
            "observations": th.zeros(
                max_steps,
                observation_embedding.obs_dim(),
                dtype=observation_tensor.dtype,
            ),
            "rewards": th.zeros(max_steps, dtype=th.float32),
            "policy_distributions": th.zeros(max_steps, n, dtype=th.float32),
            "actions": th.zeros(max_steps, dtype=th.int64),
            "mask": th.zeros(max_steps, dtype=th.bool),
            "terminals": th.zeros(max_steps, dtype=th.bool),
            "root_values": th.zeros(max_steps, dtype=th.float32),
        },
        batch_size=[max_steps],
    )

    if return_trees:
        trees = []

    tree = solver.search(env, planning_budget, observation, 0.0)

    step = 0

    while step < max_steps:

        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree)

        #print(f"Step {step}: {policy_dist.probs}")

        if return_trees:
            tree_copy = copy.deepcopy(tree)
            trees.append(tree_copy)

        distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None))

        action = distribution.sample().item()

        #print(f"Env: action = {actions_dict(env)[action]}")

        new_obs, reward, terminated, truncated, _ = env.step(action)

        if verbose:
            print(f"Env: step = {step}, obs = {print_obs(env, new_obs)}, reward = {reward}, terminated = {terminated}, truncated = {truncated}")

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

        assert not truncated

        next_terminal = terminated
        trajectory["observations"][step] = observation_tensor
        trajectory["rewards"][step] = reward
        trajectory["policy_distributions"][step] = policy_dist.probs
        trajectory["actions"][step] = action
        trajectory["mask"][step] = True
        trajectory["terminals"][step] = next_terminal

        if next_terminal or truncated:
            break

        tree = solver.search(env, planning_budget, new_obs, reward)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1

    if render:
        fps = 5
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=fps)

    if return_trees:
        return trajectory, trees

    return trajectory

@th.no_grad()
def run_episode_no_loop(
    solver: MCTS,
    env: gym.Env,
    tree_evaluation_policy: PolicyDistribution,
    observation_embedding: ObservationEmbedding,
    planning_budget=1000,
    max_steps=1000,
    seed=None,
    temperature=None,
    render=False,
    return_trees=False,
    verbose=False,
):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n = int(env.action_space.n)

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)

    observation, _ = env.reset(seed=seed)

    if verbose:
        print(f"Env: obs = {print_obs(env, observation)}")

    if render:
        vis_env = copy_environment(env)  # Use the utility function
        vis_env.unwrapped.render_mode = "rgb_array"
        frames = [vis_env.render()]

    observation_tensor: th.Tensor = observation_embedding.obs_to_tensor(observation, dtype=th.float32)

    trajectory = TensorDict(
        source={
            "observations": th.zeros(
                max_steps,
                observation_embedding.obs_dim(),
                dtype=observation_tensor.dtype,
            ),
            "rewards": th.zeros(max_steps, dtype=th.float32),
            "policy_distributions": th.zeros(max_steps, n, dtype=th.float32),
            "actions": th.zeros(max_steps, dtype=th.int64),
            "mask": th.zeros(max_steps, dtype=th.bool),
            "terminals": th.zeros(max_steps, dtype=th.bool),
            "root_values": th.zeros(max_steps, dtype=th.float32),
        },
        batch_size=[max_steps],
    )

    if return_trees:
        trees = []

    tree = solver.search(env, planning_budget, observation, 0.0)

    step = 0

    while step < max_steps:

        policy_dist = tree_evaluation_policy.softmaxed_distribution(tree, action_mask=tree.mask)

        if return_trees:
            tree_copy = copy.deepcopy(tree)
            trees.append(tree_copy)

        distribution = th.distributions.Categorical(probs=custom_softmax(policy_dist.probs, temperature, None))

        action = distribution.sample().item()

        new_obs, reward, terminated, truncated, _ = env.step(action)

        if verbose:
            print(f"Env: step = {step}, obs = {print_obs(env, new_obs)}, reward = {reward}, terminated = {terminated}, truncated = {truncated}")

        if render:
            vis_env.step(action)
            frames.append(vis_env.render())

        assert not truncated

        next_terminal = terminated
        trajectory["observations"][step] = observation_tensor
        trajectory["rewards"][step] = reward
        trajectory["policy_distributions"][step] = policy_dist.probs
        trajectory["actions"][step] = action
        trajectory["mask"][step] = True
        trajectory["terminals"][step] = next_terminal

        if next_terminal or truncated:
            break

        tree = solver.search(env, planning_budget, new_obs, reward)

        new_observation_tensor = observation_embedding.obs_to_tensor(new_obs, dtype=th.float32)
        observation_tensor = new_observation_tensor

        step += 1

    if render:
        fps = 5
        save_gif_imageio(frames, output_path=f"gifs/output.gif", fps=fps)

    if return_trees:
        return trajectory, trees

    return trajectory


