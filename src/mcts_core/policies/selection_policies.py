
import torch as th

from mcts_core.node import Node
from mcts_core.policies.policies import PolicyDistribution
from mcts_core.policies.utils import get_children_visits, get_children_mean_values

class SelectionPolicy(PolicyDistribution):

    """
    Generic parent class for selection policies in MCTS.

    Input:
    - temperature: when equal to 0, we take the argmax of the policy distribution. Otherwise we sample.

    """

    def __init__(self, *args, temperature: float = 0.0, **kwargs) -> None:
        # by default, we use argmax in selection
        super().__init__(*args, temperature=temperature, **kwargs)

class UCT(SelectionPolicy):

    """
    UCT selection policy for MCTS.
    No prior policy is used and the selection is based on the Q value of the children + an exploration term.

    Input:
    - c: parameter that determines how much we want to explore, i.e. deviate from the exploitatory Q value follow.

    """

    def __init__(self, c: float, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def Q(self, node: Node) -> th.Tensor:

        """
        Default value estimate of the children which divides 
        the total reward of the subtree by the number of visits
        """

        return get_children_mean_values(node) 

    def _probs(self, node: Node) -> th.Tensor:
        
        child_visits = get_children_visits(node)

        # if any child_visit is 0, we return 1 for all children with 0 visits
        if th.any(child_visits == 0):
            return child_visits == 0

        return self.Q(node) + self.c * th.sqrt(th.log(th.tensor(node.visits)) / child_visits)

class PUCT(UCT):

    """
    PUCT selection policy for MCTS.
    Uses prior policy to weight the exploration term.
    """

    def _probs(self, node: Node) -> th.Tensor:

        child_visits = get_children_visits(node)

        # if any child_visit is 0
        unvisited = child_visits == 0
        if th.any(unvisited):
            return node.prior_policy * unvisited

        return self.Q(node) + self.c * node.prior_policy * th.sqrt(th.tensor(node.visits)) / (child_visits + 1)

selection_dict_fn = lambda c: {
    "UCT": UCT(c, temperature=0.0),
    "PUCT": PUCT(c, temperature=0.0),
}
