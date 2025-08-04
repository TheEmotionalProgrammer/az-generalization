import torch as th
from core.node import Node

def get_children_visits(node: Node) -> th.Tensor:

    """
    Returns a tensor with the the number of visits of the children of the node.
    """

    visits = th.zeros(int(node.action_space.n), dtype=th.float32)
    for action, child in node.children.items():
        visits[action] = child.visits

    return visits

def get_mean_values(node: Node) -> th.Tensor:

    """
    Returns the value estimates of the children of the node.
    The mean estimate is used, which is the total reward of the subtree divided by the number of visits.
    """

    vals = th.ones(int(node.action_space.n), dtype=th.float32) * -th.inf 

    for action, child in node.children.items():
        vals[action] = child.mean_value()

    return vals



