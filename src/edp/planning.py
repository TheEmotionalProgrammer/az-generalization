from typing import Tuple
from gymnasium import Env
from mcts_core.node import Node
from az.planning import AlphaZeroPlanning
from mcts_core.planning import MCTS
from mcts_core.utils import observations_equal

class ExtraDeepMCTS(MCTS):

    def __init__(self, reuse_tree, block_loops, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.previous_root = None
        self.reuse_tree = reuse_tree
        self.block_loops = block_loops

    def search(self, env: Env, iterations: int, obs, reward: float) -> Node:

        root_node = self.get_root(env, obs, reward)

        counter = root_node.visits - 1
        
        while root_node.visits - counter < iterations:
            
            selected_node_for_expansion, selected_action, visited = self.traverse(root_node) # Traverse the existing tree until a leaf node is reached

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0
                self.backup(selected_node_for_expansion, 0)

            else:
                
                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                
                value = self.value_function(eval_node) # Estimate the value of the node
                        
                eval_node.value_evaluation = value # Set the value of the node

                if self.block_loops and eval_node.observation in visited:
                    eval_node.parent.mask[selected_action] = 0
                    eval_node.value_evaluation = 0.0
                    self.backup(eval_node, 0)
                else:
                    self.backup(eval_node, value) # If the parent has been masked, this will only update the visits

        return root_node # Return the root node,

    def traverse(
        self, from_node: Node
    ) -> Tuple[Node, int]:
        
        """
        Traverses the tree starting from the input node until a leaf node is reached.
        Returns the node and action to be expanded next.
        Returns None if the node is terminal.
        Note: The selection policy returns None if the input node should be expanded.
        """

        visited = set()

        node = from_node

        visited.add(node.observation)

        action = self.root_selection_policy.sample(node, mask=node.mask) # Select which node to step into

        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action, visited
                        
        node = node.step(action)  # Step into the chosen node

        visited.add(node.observation)

        while not node.is_terminal():   

            action = self.selection_policy.sample(node, mask=node.mask)

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            node = node.step(action) # Step into the chosen node

            visited.add(node.observation)

        return node, action, visited
    
    def get_root(self, env, obs, reward):
        """
        Initializes or reuses the root node based on the current state of the environment.
        """
        if self.previous_root is None or not self.reuse_tree:

            root_node = Node(
                env=env,
                parent=None,
                reward=reward,
                action_space=env.action_space,
                observation=obs,
                terminal=False,
            )

            self.previous_root = root_node

            val = self.value_function(root_node) # Estimate the value of the root node

            self.backup(root_node, val)

        else:

            root_node = self.previous_root

            found = False
            max_depth = 0
            for _, child in root_node.children.items():
                if observations_equal(child.observation, obs) and child.height > max_depth:
                    found = True
                    max_depth = child.height
                    root_node = child
                    self.previous_root = root_node

            root_node.parent = None

            if not found:
                root_node = Node(
                    env=env,
                    parent=None,
                    reward=reward,
                    action_space=env.action_space,
                    observation=obs,
                    terminal=False,

                )

                self.previous_root = root_node

                val = self.value_function(root_node) # Estimate the value of the root node

                self.backup(root_node, val)

        return root_node

class EDP(ExtraDeepMCTS, AlphaZeroPlanning):
    """
    Implementation of the EDP algorithm

    Args:
        model (AlphaZeroModel): The AlphaZero model used for value and policy prediction.
        *args: Variable length argument list.
        dir_epsilon (float, optional): The epsilon value for adding Dirichlet noise to the prior policy. Defaults to 0.0.
        dir_alpha (float, optional): The alpha value for the Dirichlet distribution. Defaults to 0.3.
        **kwargs: Arbitrary keyword arguments.
    """
    pass

