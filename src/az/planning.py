import torch as th

from mcts_core.planning import MCTS
from az.nn import AlphaZeroModel
from mcts_core.node import Node

class AlphaZeroPlanning(MCTS):

    """
    Implementation of the AlphaZero planning.

    Attributes:
        model (AlphaZeroModel): The AlphaZero model used for value and policy prediction.
        dir_epsilon (float): The epsilon value for adding Dirichlet noise to the prior policy.
        dir_alpha (float): The alpha value for the Dirichlet distribution.

    Args:
        model (AlphaZeroModel): The AlphaZero model used for value and policy prediction.
        *args: Variable length argument list.
        dir_epsilon (float, optional): The epsilon value for adding Dirichlet noise to the prior policy. Defaults to 0.0.
        dir_alpha (float, optional): The alpha value for the Dirichlet distribution. Defaults to 0.3.
        **kwargs: Arbitrary keyword arguments.
    """

    model: AlphaZeroModel
    dir_epsilon: float
    dir_alpha: float

    def __init__(
        self, model: AlphaZeroModel, *args, dir_epsilon=0.0, dir_alpha=0.3, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dir_epsilon = dir_epsilon
        self.dir_alpha = dir_alpha
                
    @th.no_grad()
    def value_function(self, node: Node) -> float:
        """
        Computes the value function for a given node.

        Args:
            node (Node): The node for which to compute the value function.

        Returns:
            float: The computed value function.
        """
        if node.is_terminal():
            return 0.0
        observation = node.observation
        # flatten the observation
        assert observation is not None
        # run the model
        # convert observation from int to tensor float 1x1 tensor
        assert node.env is not None

        value, policy = self.model.single_observation_forward(observation)

        # if root and dir_epsilon > 0.0, add dirichlet noise to the prior policy
        if node.parent is None and self.dir_epsilon > 0.0:
            if self.dir_alpha is not None:
                noise = th.distributions.dirichlet.Dirichlet(
                    th.ones_like(policy) * self.dir_alpha
                ).sample()
            else:
                # uniform distribution
                noise = th.ones_like(policy) / policy.numel()
            node.prior_policy = (
                1 - self.dir_epsilon
            ) * policy + self.dir_epsilon * noise
        else:
            node.prior_policy = policy

        return value 


