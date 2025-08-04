from collections import Counter
import torch as th

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


@th.no_grad()
def n_step_value_targets_no_reduce(
    rewards: th.Tensor,
    values: th.Tensor,
    terminals: th.Tensor,
    discount_factor: float,
    n: int,
):
    """Comutes the n-step value targets for a batch of trajectories.

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

    if n == 1:
        return one_step_value_targets(rewards, values, terminals, discount_factor)
    batch_size, n_steps = rewards.shape
    # the target should be sum_{i=0}^{n-1}(discount_factor ** i * reward_i) + discount_factor ** n * value_n
    # if we encounter a terminal state, the target is just the reward sum up to that point
    # if we reach the maximum number of steps, we sum up to that point and then add the value of the last state. This is like saying n_real = min(n, n_steps-t - 1)
    output_len = n_steps - 1
    targets = th.zeros((batch_size, output_len), device=rewards.device)
    # lets compute the target for each step
    for t in range(output_len):
        # the target for step t is the sum of the rewards up to step t + n * the value of the state at step t + n
        # we need to be careful to not go out of bounds
        n_real = min(n, output_len - t)
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


@th.no_grad()
def one_step_value_targets(
    rewards: th.Tensor, values: th.Tensor, terminals: th.Tensor, discount_factor: float
):
    targets = rewards[:, :-1] + discount_factor * values[:, 1:] * ~terminals[:, :-1]
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

