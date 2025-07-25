from collections import deque
from typing import Dict, List, Tuple
import gymnasium as gym
from gymnasium import Env
import numpy as np
from core.node import Node
from policies.policies import Policy
from core.utils import copy_environment, print_obs, observations_equal

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

        new_env = copy_environment(env) # Copy the environment
        
        root_node = Node(
            env= new_env,
            parent=None,
            reward=reward,
            action_space=env.action_space,
            observation=obs,
            #ncols=self.ncols
        )

        root_node.value_evaluation = self.value_function(root_node) # Estimate the value of the root node

        self.backup(root_node, root_node.value_evaluation) # Update node statistics

        return self.build_tree(root_node, iterations)

    def build_tree(self, from_node: Node, iterations: int) -> Node:

        """
        Builds the tree starting from the input node.
        The tree is built iteratively, and the value of the nodes is updated as the tree is built.
        """

        while from_node.visits < iterations: # Fixed number of iterations

            selected_node_for_expansion, selected_action = self.traverse(from_node) # Traverse the existing tree until a leaf node is reached

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0

                self.backup(selected_node_for_expansion, 0)

            else:

                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                value = self.value_function(eval_node) # Estimate the value of the node
                eval_node.value_evaluation = value # Set the value of the node
                self.backup(eval_node, value) # Backup the value of the node

        return from_node # Return the root node, which will now have updated statistics after the tree has been built

    def value_function(
        self,
        node: Node,
    ) -> float:
        
        """
        Depending on the specific implementation, the value of a node can be estimated in different ways.
        For this reason we leave the implementation of the value function to subclasses.
        In random rollout MCTS, the value is the sum of the future reward when acting with uniformly random policy.
        """
        
        return .0 

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
        env = copy_environment(node.env)

        assert env is not None

        # Step into the environment

        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated

        assert not truncated
        if terminated:
            observation = None

        node_class = type(node)

        # Create the node for the new state
        new_child = node_class(
            env=env,
            parent=node,
            reward=reward,
            action_space=node.action_space,
            terminal=terminal,
            observation=observation,
            action=action,
            #ncols=self.ncols
        )

        node.children[action] = new_child # Add the new node to the children of the parent node

        return new_child

    def backup(self, start_node: Node, value: float, new_visits: int = 1) -> None:
        
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
            node.visits += new_visits

            # Update the height of the node
            if node.children:
                node.height = max(child.height for child in node.children.values()) + 1
            else:
                node.height = 0  # Leaf nodes have a height of 0

            # Reset the prior policy and value evaluation (mark as needing update)

            node.policy_value = None
            node.variance = None

            node = node.parent
    
class NoLoopsMCTS(MCTS):

    def __init__(self, reuse_tree, block_loops, loops_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.previous_root = None
        self.reuse_tree = reuse_tree
        self.block_loops = block_loops
        self.loops_threshold = loops_threshold

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

        visited.add(tuple(node.observation.flatten()) if isinstance(node.observation, np.ndarray) else node.observation)

        action = self.root_selection_policy.sample(node, mask=node.mask) # Select which node to step into

        if action not in node.children: # If the selection policy returns None, this indicates that the current node should be expanded
            return node, action, visited
                        
        node = node.step(action)  # Step into the chosen node

        visited.add(tuple(node.observation.flatten()) if isinstance(node.observation, np.ndarray) else node.observation)

        while not node.is_terminal():   

            action = self.selection_policy.sample(node, mask=node.mask)

            if action not in node.children: # This means the node is not expanded, so we stop traversing the tree
                break

            node = node.step(action) # Step into the chosen node

            visited.add(tuple(node.observation.flatten()) if isinstance(node.observation, np.ndarray) else node.observation)

        return node, action, visited
    
    def expand(
        self, node: Node, action: int
    ) -> Node:
        
        """
        Expands the node and returns the expanded node.
        """

        # Copy the environment
        env = copy_environment(node.env)

        assert env is not None

        # Step into the environment

        observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated

        assert not truncated
        if terminated:
            observation = None

        node_class = type(node)

        # Create the node for the new state
        new_child = node_class(
            env=env,
            parent=node,
            reward=reward,
            action_space=node.action_space,
            terminal=terminal,
            observation=observation,
            action=action,
        )

        node.children[action] = new_child # Add the new node to the children of the parent node

        return new_child
    
    def search(self, env: Env, iterations: int, obs, reward: float, lastaction: int = None) -> Node:

        if self.previous_root is None or not self.reuse_tree:
            
            root_node = Node(
                env = env,
                parent = None,
                reward = reward,
                action_space = env.action_space,
                observation = obs,
                terminal = False,
            )

            self.previous_root = root_node

            root_node.value_evaluation = self.value_function(root_node)
            self.backup(root_node, root_node.value_evaluation)

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
                    env = env,
                    parent = None,
                    reward = reward,
                    action_space = env.action_space,
                    observation = obs,
                    terminal = False,

                )

                self.previous_root = root_node

                root_node.value_evaluation = self.value_function(root_node)
                self.backup(root_node, root_node.value_evaluation)
                        
        counter = root_node.visits 
        
        while root_node.visits - counter < iterations:
            
            selected_node_for_expansion, selected_action, visited = self.traverse(root_node) # Traverse the existing tree until a leaf node is reached

            if selected_node_for_expansion.is_terminal(): # If the node is terminal, set its value to 0 and backup
                
                selected_node_for_expansion.value_evaluation = 0.0
                self.backup(selected_node_for_expansion, 0)

            else:
                
                eval_node = self.expand(selected_node_for_expansion, selected_action) # Expand the node
                
                value = self.value_function(eval_node) # Estimate the value of the node
                        
                eval_node.value_evaluation = value # Set the value of the node

                tuple_obs = tuple(eval_node.observation.flatten()) if isinstance(eval_node.observation, np.ndarray) else eval_node.observation

                if self.block_loops and is_close_in_set(tuple_obs, visited, self.loops_threshold):
                    #print(f"Loop detected at {eval_node.observation}, blocking the action {selected_action}")
                    eval_node.parent.mask[selected_action] = 0
                    eval_node.value_evaluation = 0.0
                    self.backup(eval_node, 0)
                else:
                    self.backup(eval_node, value) # If the parent has been masked, this will only update the visits

        return root_node # Return the root node,
    
    def backup(self, start_node: Node, value: float, new_visits: int = 1) -> None:
        
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
            node.visits += new_visits

            # Update the height of the node
            if node.children:
                node.height = max(child.height for child in node.children.values()) + 1
            else:
                node.height = 0  # Leaf nodes have a height of 0

            # Reset the prior policy and value evaluation (mark as needing update)
            node.policy_value = None
            node.variance = None

            node = node.parent
                
class RandomRolloutMCTS(MCTS):
    def __init__(self, rollout_budget=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_budget = rollout_budget

    def value_function(
        self,
        node: Node,
    ) -> float:
        
        """
        The standard value function for MCTS: 
        Sum of the future reward when acting with uniformly random policy.
        """

        # if the node is terminal, return 0
        if node.is_terminal():
            return 0.0
            #return 1.0 / (1 - self.discount_factor)

        # if the node is not terminal, simulate the enviroment with random actions and return the accumulated reward until termination
        accumulated_reward = 0.0
        discount = self.discount_factor
        env = copy_environment(node.env)
        assert env is not None
        for i in range(self.rollout_budget):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            #print(print_obs(env, obs))
            accumulated_reward += reward * (discount** (i+1))
            assert not truncated
            if terminated or truncated:
                #accumulated_reward += sum(discount**(j+1) for j in range(i+1, self.rollout_budget-1))
                break

        return accumulated_reward

def compute_distances(grid_world: List[str]) -> Dict[int, int]:
    """
    Computes the distance from each cell to the goal cell, taking holes into account.
    """
    rows = len(grid_world)
    cols = len(grid_world[0])
    goal = (rows - 1, cols - 1)
    distances = {goal: 0}
    visited = set(goal)
    queue = deque([goal])

    while queue:
        row, col = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                if grid_world[new_row][new_col] == 'H':
                    continue
                visited.add((new_row, new_col))
                distances[new_row, new_col] = distances[row, col] + 1
                queue.append((new_row, new_col))
    return distances

def is_close_in_set(obs, visited, alpha=0.01):
    """
    Checks if obs is within L2 distance alpha of any element in visited.
    obs and elements of visited can be tuples or numpy arrays.
    Skips comparisons if obs or v contains None.
    """
    if obs is None or (isinstance(obs, (tuple, list, np.ndarray)) and any(x is None for x in obs)):
        return False
    obs_arr = np.array(obs)
    for v in visited:
        if v is None or (isinstance(v, (tuple, list, np.ndarray)) and any(x is None for x in v)):
            continue
        v_arr = np.array(v)
        if obs_arr.shape != v_arr.shape:
            continue
        if np.linalg.norm(obs_arr - v_arr) <= alpha:
            return True
    return False


