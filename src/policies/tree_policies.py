import torch as th
from core.node import Node
from policies.policies import PolicyDistribution
from policies.utility_functions import get_children_visits, get_children_mean_values

class VisitationPolicy(PolicyDistribution):

    """
    Visitation Counts Evaluator. 
    The action is chosen based on the number of visits to the children nodes performed during planning.
    """
    
    def _probs(self, node: Node) -> th.Tensor:
        visits = get_children_visits(node)
        return visits

class ValuePolicy(PolicyDistribution):

    """
    Value Evaluator. 
    The action is chosen based on the value estimates of the children nodes.
    """

    def _probs(self, node: Node) -> th.Tensor:
        values = get_children_mean_values(node)
        return values

tree_eval_dict = lambda temperature=None: {
    "visit": VisitationPolicy(temperature),
    "q": ValuePolicy(temperature)
}
