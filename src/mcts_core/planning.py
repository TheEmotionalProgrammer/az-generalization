from typing import Tuple
import gymnasium as gym
from mcts_core.node import Node
from mcts_core.policies.policies import Policy
from copy import deepcopy

class MCTS:

    """
    This class contains the basic MCTS algorithm without assumptions on the leafs value estimation.
    """

    root_selection_policy: Policy
    selection_policy: Policy

    def __init__(
        self,
        selection_policy: Policy,
        discount_factor: float = 1.0,
        root_selection_policy: Policy | None = None,
    ):
        if root_selection_policy is None:
            root_selection_policy = selection_policy
        self.root_selection_policy = root_selection_policy
        self.selection_policy = selection_policy  # the selection policy should return None if the input node should be expanded
        self.discount_factor = discount_factor

    def search(
        self,
        env: gym.Env,
        iterations: int,
        obs,
        reward: float,
    ) -> Node:
        
        """
        The main function of the MCTS algorithm. Returns the current root node with updated statistics.
        It builds a tree of nodes starting from the root, i.e. the current state of the environment.
        The tree is built iteratively, and the value of the nodes is updated as the tree is built.
        """
        
        assert isinstance(env.action_space, gym.spaces.Discrete) # Assert that the type of the action space is discrete

        new_env = deepcopy(env) # Copy the environment
        
        root_node = Node(
            env= new_env,
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
        )

        val = self.value_function(root_node) # Estimate the value of the root node

        self.backup(root_node, val) # Initialize node visits and height

        while root_node.visits < iterations: # Fixed number of iterations

            selected_node_for_expansion, selected_action = self.traverse(root_node) # Traverse the existing tree until a leaf node is reached

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0

                self.backup(selected_node_for_expansion, 0)

            else:

                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                value = self.value_function(eval_node) # Estimate the value of the node
                eval_node.value_evaluation = value # Set the value of the node
                self.backup(eval_node, value) # Backup the value of the node

        return root_node # Return the root node, which will now have updated statistics after the tree has been built

    def traverse(
        self, from_node: Node
    ) -> Tuple[Node, int]:
        
        """
        Traverses the tree starting from the input node until a leaf node is reached.
        Returns the node and action to be expanded next.
        Returns None if the node is terminal.
        Note: The selection policy returns None if the input node should be expanded.
        """

        node = from_node

        action = self.root_selection_policy.sample(node) # Select which node to step into
        
        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action
        
        node = node.step(action)  # Step into the chosen node

        while not node.is_terminal():
            
            action = self.selection_policy.sample(node) # Select which node to step into

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            node = node.step(action) # Step into the chosen node
            
        return node, action

    def expand(
        self, node: Node, action: int
    ) -> Node:
        
        """
        Expands the node and returns the expanded node.
        """

        # Copy the environment
        env = deepcopy(node.env)

        # Step into the environment
        observation, reward, terminal, _, _ = env.step(action)

        if terminal:
            observation = None

        # Create the node for the new state
        new_child = Node(
            env=env,
            parent=node,
            reward=reward,
            action_space=node.action_space,
            terminal=terminal,
            observation=observation,
        )

        node.children[action] = new_child # Add the new node to the children of the parent node

        return new_child

    def backup(self, start_node: Node, value: float) -> None:
        
        """
        Backups the value of the start node to its parent, grandparent, etc., all the way to the root node.
        Updates the statistic of the nodes in the path:
        - subtree_sum: the sum of the value of the node and its children
        - visits: the number of times the node has been visited
        - height: the maximum number of steps from the node to a leaf
        """

        node = start_node
        cumulative_reward = value

        while node is not None: # The parent is None if node is the root
            
            cumulative_reward *= self.discount_factor
            cumulative_reward += node.reward
            node.subtree_sum += cumulative_reward
            node.visits += 1

            # Update the height of the node
            if node.children:
                node.height = max(child.height for child in node.children.values()) + 1
            else:
                node.height = 0  # Leaf nodes have a height of 0

            node = node.parent

    def value_function(
        self,
        node: Node,
    ) -> float:
        
        """
        Depending on the specific implementation, the value of a node can be estimated in different ways.
        For this reason we leave the implementation of the value function to subclasses.
        In random rollout MCTS, the value is the sum of the future reward when acting with uniformly random policy.
        In AlphaZero MCTS, the value is estimated with a neural network.
        """
        
        raise NotImplementedError("The value function has to be implemented in subclasses of MCTS.")
    

    
    




