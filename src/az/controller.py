import os
from collections import Counter
from typing import List

import gymnasium as gym
import numpy as np
import torch as th
import wandb
from environments.observation_embeddings import CoordinateEmbedding
from torchrl.data import TensorDictReplayBuffer
from tqdm import tqdm

from az.planning import AlphaZeroPlanning
from mcts_core.runner import collect_trajectories
from mcts_core.runner import eval_agent
from utils.metrics import calc_metrics
from utils.wandb_logs import (
    add_self_play_metrics_wandb,
    add_training_metrics_wandb,
    plot_visits_to_wandb_with_counter,
    show_model_in_wandb,
)
from mcts_core.policies.policies import PolicyDistribution
from mcts_core.policies.tree_policies import VisitationPolicy

class AlphaZeroController:
    
    """
    The Controller will be responsible for orchestrating the training of the model. With self play and training.
    """

    def __init__(
        self,
        env: gym.Env,
        agent: AlphaZeroPlanning,
        optimizer: th.optim.Optimizer,
        replay_buffer=TensorDictReplayBuffer(),
        training_epochs=10,
        tree_evaluation_policy: PolicyDistribution = VisitationPolicy(),
        planning_budget=100,
        max_episode_length=500,
        run_dir="./logs",
        checkpoint_interval=-1,  # -1 means no checkpoints
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        reg_loss_weight=0.0,
        episodes_per_iteration=10,
        self_play_workers=1,
        scheduler: th.optim.lr_scheduler.LRScheduler | None = None,
        value_sim_loss=False,
        discount_factor=1.0,
        n_steps_learning: int = 1,
        use_visit_count=False,
        save_plots=True,
        batch_size=32,
        ema_beta=0.3,
        evaluation_interval=10,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.training_epochs = training_epochs
        self.optimizer = optimizer
        self.agent = agent
        self.env = env
        self.tree_evaluation_policy = tree_evaluation_policy
        self.planning_budget = planning_budget
        self.max_episode_length = max_episode_length
        self.self_play_workers = self_play_workers
        self.run_dir = run_dir
        self.value_sim_loss = value_sim_loss
        # create run dir if it does not exist
        os.makedirs(self.run_dir, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self.discount_factor = discount_factor
        self.n_steps_learning = n_steps_learning
        # Log the model

        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.reg_loss_weight = reg_loss_weight 
        self.episodes_per_iteration = episodes_per_iteration
        self.scheduler = scheduler
        self.train_obs_counter = Counter()
        self.use_visit_count = use_visit_count
        self.save_plots = save_plots
        self.batch_size = batch_size
        self.ema_beta = ema_beta
        self.evaluation_interval = evaluation_interval
        self.best_mean_return = float('-inf')  # Initialize to negative infinity
        self.best_mean_discounted_return = float('-inf')  # Initialize to negative infinity

    def iterate(self, temp_schedule: List[float], seed=None):
        """
        Perform iterations of self-play, learning, and evaluation.

        Args:
            temp_schedule (List[float]): A list of temperature values for self-play.

        Returns:
            dict: A dictionary containing the average return over iterations.
        """
        total_return = 0.0
        enviroment_steps = 0
        episodes = 0
        ema = None
        iterations = len(temp_schedule)
        for i, temperature in enumerate(temp_schedule):
            print(f"Iteration {i}")
            print("Self play...")
            tensor_results = self.self_play(temperature=temperature, seed=seed)
            self.replay_buffer.extend(tensor_results)
            total_return, ema = self.add_self_play_metrics(
                tensor_results, total_return, ema, i
            )
            print("Learning...")
            (
                value_losses,
                policy_losses,
                total_losses,
                value_sims,
            ) = self.learn() 

            # the regularization loss is the squared l2 norm of the weights
            regularization_loss = th.tensor(0.0, device=self.agent.model.device)
            for param in self.agent.model.parameters():
                regularization_loss += th.sum(th.square(param))

            add_training_metrics_wandb(
                value_losses,
                policy_losses,
                value_sims,
                regularization_loss,
                total_losses,
                len(self.replay_buffer),
                self.scheduler.get_last_lr() if self.scheduler else None,
                i,
            )

            # if self.checkpoint_interval != -1 and i % self.checkpoint_interval == 0:
            #     print(f"Saving model at iteration {i}")
            #     self.agent.model.save_model(f"{self.run_dir}/checkpoint.pth")

            if self.scheduler is not None:
                self.scheduler.step()

            # if the env is CliffWalking-v0, plot the output of the value and policy networks
            assert self.env.spec is not None

            if (
                type(self.agent.model.observation_embedding) == CoordinateEmbedding and self.save_plots
            ):
                assert isinstance(self.env.observation_space, gym.spaces.Discrete)

                desc = self.env.unwrapped.desc

                # wandb
                show_model_in_wandb(self.agent.model, i, desc)
                plot_visits_to_wandb_with_counter(
                    self.train_obs_counter, 
                    self.agent.model.observation_embedding, 
                    i, 
                    desc
                )
            
            time_steps = tensor_results["mask"].sum(dim=-1)
            enviroment_steps += th.sum(time_steps).item()
            episodes += time_steps.shape[0]
            wandb.log(
                {
                    "environment_steps": enviroment_steps,
                    "episodes": episodes,
                    "grad_steps": i * self.training_epochs,
                },
                step=i,
            )

            if i % self.evaluation_interval == 0 or i == iterations - 1:
                print("Evaluating...")
                self.evaluate(i, start_seed=seed)

        # if self.checkpoint_interval != -1:
        #     print(f"Saving model at iteration {iterations}")
        #     self.agent.model.save_model(f"{self.run_dir}/checkpoint.pth")

        return {"average_return": total_return / iterations}

    def evaluate(self, step: int, start_seed=None):
        """
        Evaluate the agent's performance by collecting one trajectory with a non-stochastic version of the policy.

        Args:
            step (int): The current step or iteration number.

        Returns:
            dict: A dictionary containing evaluation metrics and trajectories.

        """
        # collect one trajectory with non-stochastic version of the policy
        self.agent.model.eval()
        # temporarly set the epsilon to 0
        # eps, self.agent.dir_epsilon = self.agent.dir_epsilon, 0.0
        alpha, self.agent.dir_alpha = self.agent.dir_alpha, None

        seeds = (
            [start_seed * self.episodes_per_iteration + i for i in range(self.episodes_per_iteration)]
            if start_seed is not None
            else [None] * self.episodes_per_iteration
        )

        results = eval_agent(
            self.agent,
            self.env,
            self.tree_evaluation_policy,
            self.agent.model.observation_embedding,
            self.planning_budget,
            self.max_episode_length,
            seeds=seeds,
            temperature=0.0,
        )

        episode_returns, discounted_returns, time_steps, entropies = calc_metrics(
            results, self.agent.discount_factor, self.env.action_space.n
        )
        # apply the observation embedding to the last dimension of the observations tensor
        trajectories = []
        for i in range(results.shape[0]):
            re = []
            for j in range(results.shape[1]):
                re.append(
                    self.agent.model.observation_embedding.tensor_to_obs(
                        results[i, j]["observations"]
                    )
                )
                if results[i, j]["terminals"] == 1:
                    break
            trajectories.append(re)

        eval_res = {
            "Evaluation/Mean_Returns": episode_returns.mean().item(),
            "Evaluation/Mean_Discounted_Returns": discounted_returns.mean().item(),
            "Evaluation/Mean_Timesteps": time_steps.mean().item(),
            "Evaluation/Mean_Discounted_Returns": discounted_returns.mean().item(),
            "Evaluation/Mean_Entropy": (th.sum(entropies, dim=-1) / time_steps).mean().item(),
        }
        wandb.log(
            eval_res,
            step=step,
        )

        # Save the model if the current mean discounted return is better than the best so far

        current_mean_discounted_return = eval_res["Evaluation/Mean_Discounted_Returns"]

        if current_mean_discounted_return >= self.best_mean_discounted_return:
            print(f"New best mean discounted return: {current_mean_discounted_return}. Saving model...")
            self.best_mean_discounted_return = current_mean_discounted_return
            self.agent.model.save_model(f"{self.run_dir}/checkpoint.pth")

        # self.agent.dir_epsilon = eps
        self.agent.dir_alpha = alpha
        return eval_res

    def self_play(self, temperature=None, seed=None):
        """Play games in parallel and store the data in the replay buffer."""
        self.agent.model.eval()

        #Generate unique seeds for each task if a seed is provided
        seeds = (
            [seed * self.episodes_per_iteration + i for i in range(self.episodes_per_iteration)]
            if seed is not None
            else [None] * self.episodes_per_iteration
        )

        tasks = [
            (
                self.agent,
                self.env,
                self.tree_evaluation_policy,
                self.agent.model.observation_embedding,
                self.planning_budget,
                self.max_episode_length,
                task_seed,
                temperature,
            )
            for task_seed in seeds
        ]

        # Create tasks with unique seeds


        return collect_trajectories(tasks, self.self_play_workers)

    def add_self_play_metrics(self, tensor_res, total_return, last_ema, global_step):
        """
        Adds self-play metrics to the AlphaZero class.

        Args:
            tensor_res (Tensor): The tensor result.
            total_return (float): The total return.
            last_ema (float): The last exponential moving average.
            global_step (int): The global step.

        Returns:
            tuple: A tuple containing the updated total return and exponential moving average.
        """
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        episode_returns, discounted_returns, time_steps, entropies = calc_metrics(
            tensor_res, self.discount_factor, self.env.action_space.n
        )

        mean_entropies = th.sum(entropies, dim=-1) / time_steps
        # Calculate statistics
        mean_return = th.mean(discounted_returns).item()
        # return_variance = th.var(mean_return, ddof=1)
        total_return += mean_return
        if last_ema is None:
            last_ema = mean_return
        ema_return = mean_return * self.ema_beta + last_ema * (1 - self.ema_beta)

        add_self_play_metrics_wandb(
            np.array(episode_returns),
            np.array(discounted_returns),
            np.array(time_steps),
            np.array(mean_entropies),
            total_return,
            ema_return,
            global_step,
        )

        return total_return, ema_return

    def learn(self):
        """
        Perform the learning process of the AlphaZero algorithm.

        Returns:
            value_losses (list): List of value losses for each training epoch.
            policy_losses (list): List of policy losses for each training epoch.
            total_losses (list): List of total losses (value loss + policy loss + regularization loss) for each training epoch.
            value_sims (list): List of value similarities for each training epoch.
        """
        value_losses = []
        policy_losses = []
        total_losses = []
        value_sims = []
        self.agent.model.train()
        for _ in tqdm(range(self.training_epochs), desc="Training"):
            # sample a batch from the replay buffer
            with th.no_grad():
                trajectories = self.replay_buffer.sample(
                    batch_size=min(self.batch_size, len(self.replay_buffer))
                )
                # Trajectories["observations"] is Batch_size x max_steps x obs_dim
                observations = trajectories["observations"]
                batch_size, max_steps, obs_dim = observations.shape

                # flatten the observations into a batch of size (batch_size * max_steps, obs_dim)
                flattened_observations = observations.view(-1, obs_dim)
                epsilon = 1e-8

            flat_values, flat_policies = self.agent.model.forward(
                flattened_observations
            )
            values = flat_values.view(batch_size, max_steps)
            policies = flat_policies.view(batch_size, max_steps, -1)

            # compute the value targets via TD learning
            with th.no_grad():
                value_simularities = th.exp(
                    -th.sum(
                        (
                            trajectories["mask"]
                            * (1 - trajectories["root_values"] / values.detach())
                        )
                        ** 2,
                        dim=-1,
                    )
                    / trajectories["mask"].sum(dim=-1)
                )

                norm_visit_multiplier = th.ones_like(values)
                if self.use_visit_count or self.save_plots:
                    visit_count_tensor, counter = calculate_visit_counts(
                        observations, trajectories["mask"]
                    )
                    self.train_obs_counter.update(counter)
                    if self.use_visit_count:
                        visit_multiplier = trajectories["mask"] / (
                            visit_count_tensor + epsilon
                        )
                        norm_visit_multiplier = visit_multiplier * (
                            trajectories["mask"].sum() / visit_multiplier.sum()
                        )

                targets = n_step_value_targets(
                    trajectories["rewards"],
                    values.detach(),
                    trajectories["terminals"],
                    self.discount_factor,
                    self.n_steps_learning,
                )
                dim_red = self.n_steps_learning
                mask = trajectories["mask"][:, :-dim_red]

            td = targets - values[:, :-dim_red]
            step_loss = (td * mask) ** 2 * norm_visit_multiplier[:, :-dim_red]
            if self.value_sim_loss:
                value_loss = th.sum(
                    th.sum(step_loss, dim=-1) * value_simularities
                ) / th.sum(mask)
            else:
                value_loss = th.sum(step_loss) / th.sum(mask)

            step_loss = -th.einsum(
                "ijk,ijk->ij",
                trajectories["policy_distributions"],
                th.log(policies + epsilon),
            )
            policy_loss = th.sum(
                step_loss * trajectories["mask"] * norm_visit_multiplier
            ) / th.sum(trajectories["mask"])

            # Compute the regularization loss
            regularization_loss = th.tensor(0.0, device=self.agent.model.device)
            for param in self.agent.model.parameters():
                regularization_loss += th.sum(th.square(param))

            # Include the regularization loss in the total loss
            loss = (
                self.value_loss_weight * value_loss
                + self.policy_loss_weight * policy_loss
                + self.reg_loss_weight * regularization_loss
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())
            total_losses.append(loss.item())
            value_sims.append(value_simularities.mean().item())

        return value_losses, policy_losses, total_losses, value_sims

@th.no_grad()
def n_step_value_targets(
    rewards: th.Tensor,
    values: th.Tensor,
    terminals: th.Tensor,
    discount_factor: float,
    n: int,
):
    """Computes the n-step value targets for a batch of trajectories.
    Returns only true n-step targets, not the intermediate steps.

    Args:
        rewards (th.Tensor): reward for step t+1
        values (th.Tensor): value of state t
        terminals (th.Tensor): terminal at step t+1
        mask (th.Tensor): step t valid
        discount_factor (float): discount factor
        n (int): number of steps to look ahead

    Returns:
        targets (th.Tensor): targets for the value function dimension (batch_size, n_steps-1)

    """
    assert n > 0

    batch_size, n_steps = rewards.shape
    # the target should be sum_{i=0}^{n-1}(discount_factor ** i * reward_i) + discount_factor ** n * value_n
    # if we encounter a terminal state, the target is just the reward sum up to that point
    # if we reach the maximum number of steps, we sum up to that point and then add the value of the last state. This is like saying n_real = min(n, n_steps-t - 1)
    output_len = n_steps - n
    targets = th.zeros((batch_size, output_len), device=rewards.device)
    # lets compute the target for each step
    for t in range(output_len):
        # the target for step t is the sum of the rewards up to step t + n * the value of the state at step t + n
        # we need to be careful to not go out of bounds
        n_real = n
        targets[:, t] = (
            th.sum(
                rewards[:, t : t + n_real] * discount_factor ** th.arange(n_real),
                dim=-1,
            )
            + discount_factor**n_real
            * values[:, t + n_real]
            * ~terminals[:, t:t + n_real].any(dim=-1) # There were previously a bug here where the value of states after terminals were still counted
        )

    return targets

def calculate_visit_counts(observations, full_mask):
    """
    Calculate the visit counts for each observation in the trajectories, with output shape (batch_size, max_steps).
    Observations are assumed to be a tensor of shape (batch_size, max_steps, obs_dim).
    """

    batch_size, max_steps, obs_dim = observations.shape

    # Create a tensor to hold the visit counts for each observation, initially all zeros
    visit_counts = th.zeros(batch_size, max_steps)

    visit_counts_mapping = Counter()

    for b in range(batch_size):
        for t in range(max_steps):
            # break on mask
            if not full_mask[b, t]:
                break
            current_observation = observations[b, t]
            obs_key = tuple(current_observation.tolist())
            visit_counts_mapping[obs_key] += 1

    for b in range(batch_size):
        for t in range(max_steps):
            # break on mask
            if not full_mask[b, t]:
                break
            current_observation = observations[b, t]
            obs_key = tuple(current_observation.tolist())
            visit_counts[b, t] = visit_counts_mapping[obs_key]


    return visit_counts, visit_counts_mapping