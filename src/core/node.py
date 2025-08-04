from collections import Counter
from typing import Dict, Generic, List, TypeVar, Optional, Any, Callable
import gymnasium as gym
import numpy as np
import torch as th
import graphviz
from core.utils import actions_dict, print_obs
from environments.gridworld.grid_world import GridWorldEnv

ObservationType = TypeVar("ObservationType")

NodeType = TypeVar("NodeType", bound="Node")

class Node(Generic[ObservationType]):

    parent: Optional["Node[ObservationType]"]
    children: Dict[int, "Node[ObservationType]"]
    visits: int = 0
    subtree_sum: float = 0.0  # sum of reward and value of all children
    value_evaluation: float  # Expected future reward
    reward: float  # Reward received when stepping into this node
    action_space: gym.spaces.Discrete  
    observation: Optional[ObservationType]
    prior_policy: th.Tensor
    env: Optional[gym.Env]

    def __init__(
        self,
        env: gym.Env,
        parent: Optional["Node[ObservationType]"],
        reward: float,
        action_space: gym.spaces.Discrete,
        observation: Optional[ObservationType],
        terminal: bool = False,
    
    ):
        
        self.children = {} # dictionary of children where key is action, value is the child node
        self.action_space = action_space
        self.mask = np.ones(self.action_space.n, dtype=np.int8)
        self.reward = reward
        self.parent = parent
        self.terminal = terminal
        self.observation = observation
        self.env = env
        self.height = 0
        self.ncols = None if not isinstance(env, GridWorldEnv) else env.unwrapped.ncol

    def is_terminal(self) -> bool:
        return self.terminal
                
    def step(self, action: int) -> "Node[ObservationType]":

        """
        Return the child node after taking the given action. 
        The child has to be expanded already, otherwise the method will throw an error.
        """
        
        child = self.children[action]
        return child

    def mean_value(self) -> float:

        """
        The mean value estimate for taking this action is the average of the rewards + value estimates of all children
        """

        return self.subtree_sum / self.visits

    def is_fully_expanded(self) -> bool:

        """
        Returns True if all possible actions have been expanded.
        """

        return len(self.children) == self.action_space.n

    def visualize(
        self,
        var_fn: Optional[Callable[["Node[ObservationType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        
        dot = graphviz.Digraph(comment="Planning Tree")
        self._add_node_to_graph(dot, var_fn, max_depth=max_depth)
        dot.render(filename="plan_tree.gv", view=True)

    def _add_node_to_graph(
        self,
        dot,
        var_fn: Optional[Callable[["Node[ObservationType]"], Any]] = None,
        max_depth: Optional[int] = None,
    ) -> None:
        if max_depth is not None and max_depth == 0:
            return
        label = f"O: {print_obs(self.env, self.observation)}, V: {self.value_evaluation: .2f}, Visit: {self.visits}, \nT: {int(self.terminal)}, R: {self.reward}, P: {self.problematic}"
        if var_fn is not None:
            label += f", VarFn: {var_fn(self)}"

        dot.node(str(id(self)), label=label)
        for action, child in self.children.items():
            child._add_node_to_graph(
                dot, var_fn, max_depth=max_depth - 1 if max_depth is not None else None
            )

            dot.edge(str(id(self)), str(id(child)), label=f"Action: {actions_dict(self.env)[action]}")


    def state_visitation_counts(self) -> Counter:

        """
        Returns a counter of the number of times each state has been visited
        """

        counter = Counter()
        # add the current node
        counter[self.observation] = self.visits #if not self.is_terminal() else 1
        # add all children
        for child in self.children.values():
            counter.update(child.state_visitation_counts())

        return counter
    
    def state_repetitions(self) -> Counter:
        """
        Returns a counter of the number of times each state appears in the tree (in different nodes)"""
        counter = Counter()
        # add the current node
        counter[self.observation] = 1
        # add all children
        for child in self.children.values():
            counter.update(child.state_repetitions())

        return counter
    
    def mean_nn_values_dict(self) -> Dict[int, float]:
        """
        Returns a dictionary with the observations as keys and the mean value estimates
        of the subtrees as values, recursively including all descendants.
        """
        values = {}
        counts = {}

        # Initialize with the current node's observation and value
        if self.observation is not None:
            values[self.observation] = self.value_evaluation
            counts[self.observation] = 1

        # Recursively process children
        for child in self.children.values():
            if child.observation is not None:
                # Get the child's mean_nn_values_dict recursively
                child_values = child.mean_nn_values_dict()
                for obs, value in child_values.items():
                    if obs not in values:
                        values[obs] = value
                        counts[obs] = 1
                    else:
                        values[obs] += value
                        counts[obs] += 1

        # Compute the mean for each observation
        for key in values.keys():
            values[key] /= counts[key]

        return values
    
    def sum_nn_values_dict(self) -> Dict[int, float]:
        """
        Returns a dictionary with the observations as keys and the sum of value estimates
        of the subtrees as values, recursively including all descendants.
        """
        values = {}

        # Initialize with the current node's observation and value
        if self.observation is not None:
            values[self.observation] = self.value_evaluation

        # Recursively process children
        for child in self.children.values():
            if child.observation is not None:
                # Get the child's nn_sum_values_dict recursively
                child_values = child.sum_nn_values_dict()
                for obs, value in child_values.items():
                    if obs not in values:
                        values[obs] = value
                    else:
                        values[obs] += value

        return values
            
    def get_children(self):

        """
        Returns the list of children of the node.
        """

        l: List[Node | None] = [None] * self.action_space.n
        for key, child in self.children.items():
            l[key] = child
        return l

    def __str__(self):
        return f"Visits: {self.visits}, ter: {int(self.terminal)}\nR: {self.reward}\n Value_Estimate: {self.value_evaluation}, Mean_Value: {self.default_value()}"

    def __repr__(self):
        return self.__str__()
