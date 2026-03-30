"""
Maze Environment Wrapper

Wraps MazeGraph from corridors.py into a standard RL environment interface.
Handles state/goal representation and reward structure.
"""

import numpy as np
import networkx as nx
from corridors import MazeGraph


class MazeEnvironment:
    def __init__(self, length=8, width=8, corridor=0.5, seed=None, control_cost=0.01,
                 fixed_start_node=None, goal_is_deadend=False):
        """
        Initialize maze environment.

        Parameters:
        -----------
        length : int
            Height of the maze grid
        width : int
            Width of the maze grid
        corridor : float
            Corridor parameter (0=junctions, 1=corridors)
        seed : int, optional
            Random seed for reproducibility
        control_cost : float
            Cost per step for using slow processing
        fixed_start_node : tuple, optional
            Fixed starting position (row, col) for all episodes.
            If None, start position is randomized each episode.
        goal_is_deadend : bool
            If True, goal is always selected from dead-end nodes (degree 1).
            If False, goal is selected randomly from all nodes.
        """
        self.length = length
        self.width = width
        self.corridor = corridor
        self.seed = seed
        self.control_cost = control_cost
        self.fixed_start_node = fixed_start_node  # None = random, else fixed
        self.goal_is_deadend = goal_is_deadend

        # Generate maze graph
        self.maze = MazeGraph(length=length, width=width, corridor=corridor, seed=seed)
        self.graph = self.maze.get_graph()

        # Create node mappings
        self.nodes_list = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        self.num_nodes = len(self.nodes_list)
        self.num_actions = 5  # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL

        # Direction mappings
        self.DIRECTION_UP = 0
        self.DIRECTION_DOWN = 1
        self.DIRECTION_LEFT = 2
        self.DIRECTION_RIGHT = 3
        self.IDENTIFY_GOAL = 4

        # Identify dead-end nodes (degree 1)
        self.deadend_nodes = [node for node in self.nodes_list if self.graph.degree(node) == 1]
        if len(self.deadend_nodes) == 0:
            print("WARNING: No dead-end nodes found in maze. Using all nodes for goal selection.")
            self.deadend_nodes = self.nodes_list.copy()

        # State variables
        self.current_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.max_steps = self.num_nodes * 2  # Maximum episode length

        # Statistics
        self.total_control_cost = 0.0
        self.used_slow_count = 0

    def reset(self, start_pos=None, goal_pos=None):
        """
        Reset environment to initial state.

        Parameters:
        -----------
        start_pos : tuple, optional
            Starting position (row, col). If provided, overrides fixed_start_node.
            If None and fixed_start_node is set, uses fixed_start_node.
            If None and fixed_start_node is None, randomizes start position.
        goal_pos : tuple, optional
            Goal position (row, col). If None, random based on goal_is_deadend setting.

        Returns:
        --------
        state : dict
            Initial state with current_pos, goal_pos, and step_count
        """
        # Set start position
        if start_pos is not None:
            # Explicit override
            self.current_pos = start_pos
        elif self.fixed_start_node is not None:
            # Use fixed start node
            self.current_pos = self.fixed_start_node
        else:
            # Random start (fixed_start_node is None)
            self.current_pos = self.nodes_list[np.random.randint(self.num_nodes)]

        # Verify start position is valid
        if self.current_pos not in self.nodes_list:
            raise ValueError(f"Start position {self.current_pos} is not a valid node in the maze")

        # Set goal position
        if goal_pos is None:
            # Choose goal based on goal_is_deadend setting
            if self.goal_is_deadend:
                # Select from dead-end nodes only
                possible_goals = [n for n in self.deadend_nodes if n != self.current_pos]
                if len(possible_goals) == 0:
                    # If start is the only dead-end, select from all other nodes
                    possible_goals = [n for n in self.nodes_list if n != self.current_pos]
            else:
                # Select from all nodes
                possible_goals = [n for n in self.nodes_list if n != self.current_pos]

            if len(possible_goals) == 0:
                raise ValueError(f"No valid goal positions (maze has only one node)")

            self.goal_pos = possible_goals[np.random.randint(len(possible_goals))]
        else:
            self.goal_pos = goal_pos

        self.step_count = 0
        self.total_control_cost = 0.0
        self.used_slow_count = 0

        return self._get_state()

    def _get_state(self):
        """
        Get current state representation.

        Returns:
        --------
        state : dict
            Dictionary containing:
            - current_pos: current position tuple
            - goal_pos: goal position tuple
            - current_idx: index of current position
            - goal_idx: index of goal position
            - current_encoding: one-hot encoding of current position
            - goal_encoding: one-hot encoding of goal position
            - step_count: number of steps taken
        """
        current_encoding = np.zeros(self.num_nodes, dtype=np.float32)
        current_encoding[self.node_to_idx[self.current_pos]] = 1.0

        goal_encoding = np.zeros(self.num_nodes, dtype=np.float32)
        goal_encoding[self.node_to_idx[self.goal_pos]] = 1.0

        return {
            'current_pos': self.current_pos,
            'goal_pos': self.goal_pos,
            'current_idx': self.node_to_idx[self.current_pos],
            'goal_idx': self.node_to_idx[self.goal_pos],
            'current_encoding': current_encoding,
            'goal_encoding': goal_encoding,
            'step_count': self.step_count
        }

    def step(self, action, used_slow=False):
        """
        Take a step in the environment.

        Parameters:
        -----------
        action : int
            Action in direction space:
            0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL
        used_slow : bool
            Whether slow processing was used (for control cost)

        Returns:
        --------
        next_state : dict
            Next state representation
        reward : float
            Reward received
        done : bool
            Whether episode is complete
        info : dict
            Additional information
        """
        invalid_move = False
        old_pos = self.current_pos

        # Handle IDENTIFY_GOAL action
        if action == self.IDENTIFY_GOAL:
            if self.current_pos == self.goal_pos:
                # Correctly identified goal
                reward = 10.0
                done = True
            else:
                # Incorrectly identified goal - stay in place
                reward = -0.1  # Standard step penalty
                done = False
                invalid_move = False  # Not invalid, just incorrect

        # Handle direction-based movement
        elif action in [self.DIRECTION_UP, self.DIRECTION_DOWN, self.DIRECTION_LEFT, self.DIRECTION_RIGHT]:
            # Get valid neighbors in all directions
            direction_neighbors = self.maze.get_direction_neighbors(self.current_pos)

            # Check if this direction is valid
            if action in direction_neighbors:
                # Valid move - update position
                self.current_pos = direction_neighbors[action]
                invalid_move = False
            else:
                # Invalid move - stay in place
                invalid_move = True

            # Calculate reward (only if not identify_goal)
            reward = -0.1  # Standard step penalty
            done = False

        else:
            # Invalid action value
            invalid_move = True
            reward = -0.1
            done = False

        self.step_count += 1

        # Control cost
        if used_slow:
            reward -= self.control_cost
            self.total_control_cost += self.control_cost
            self.used_slow_count += 1

        # Timeout
        if self.step_count >= self.max_steps:
            done = True
            reward -= 5.0  # Penalty for timeout

        info = {
            'invalid_move': invalid_move,
            'used_slow': used_slow,
            'total_control_cost': self.total_control_cost,
            'used_slow_count': self.used_slow_count,
            'reached_goal': self.current_pos == self.goal_pos,
            'old_pos': old_pos
        }

        return self._get_state(), reward, done, info

    def get_optimal_path_length(self, start=None, goal=None):
        """
        Get length of optimal path from start to goal.

        Returns -1 if no path exists.
        """
        if start is None:
            start = self.current_pos
        if goal is None:
            goal = self.goal_pos

        if not nx.has_path(self.graph, start, goal):
            return -1

        path = nx.shortest_path(self.graph, start, goal)
        return len(path) - 1  # Number of steps

    def get_optimal_next_action(self, pos=None, goal=None):
        """
        Get optimal next action (direction) from pos toward goal using shortest path.

        Returns None if no path exists. Returns IDENTIFY_GOAL if already at goal.
        """
        if pos is None:
            pos = self.current_pos
        if goal is None:
            goal = self.goal_pos

        # If at goal, identify it
        if pos == goal:
            return self.IDENTIFY_GOAL

        if not nx.has_path(self.graph, pos, goal):
            return None

        path = nx.shortest_path(self.graph, pos, goal)
        if len(path) < 2:
            return self.IDENTIFY_GOAL

        next_pos = path[1]

        # Convert next position to direction
        r_curr, c_curr = pos
        r_next, c_next = next_pos

        dr = r_next - r_curr
        dc = c_next - c_curr

        # Map to direction
        if dr == -1 and dc == 0:
            return self.DIRECTION_UP
        elif dr == 1 and dc == 0:
            return self.DIRECTION_DOWN
        elif dr == 0 and dc == -1:
            return self.DIRECTION_LEFT
        elif dr == 0 and dc == 1:
            return self.DIRECTION_RIGHT
        else:
            # Should not happen if graph is valid
            return None

    def render(self, title=None):
        """
        Visualize current state of the environment.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('#F8F9FA')

        # Draw maze structure
        pos = {(r, c): (c + 0.5, r + 0.5) for r in range(self.length) for c in range(self.width)}

        # Draw edges
        for edge in self.graph.edges():
            node1, node2 = edge
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]
            ax.plot([x1, x2], [y1, y2], color='#457B9D', linewidth=2.5, alpha=0.6, zorder=1)

        # Draw regular nodes
        for node in self.graph.nodes():
            if node not in [self.current_pos, self.goal_pos]:
                x, y = pos[node]
                ax.scatter(x, y, s=100, c='#A8DADC', alpha=0.8, edgecolors='white', linewidths=1.5, zorder=2)

        # Draw current position
        x, y = pos[self.current_pos]
        ax.scatter(x, y, s=300, c='#E63946', alpha=0.9, edgecolors='white', linewidths=2, zorder=3, marker='o')
        ax.text(x, y, 'S', ha='center', va='center', fontsize=16, fontweight='bold', color='white', zorder=4)

        # Draw goal position
        x, y = pos[self.goal_pos]
        ax.scatter(x, y, s=300, c='#2A9D8F', alpha=0.9, edgecolors='white', linewidths=2, zorder=3, marker='*')
        ax.text(x, y-0.3, 'G', ha='center', va='center', fontsize=16, fontweight='bold', color='white', zorder=4)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        if title is None:
            title = f'Step {self.step_count} | Goal distance: {self.get_optimal_path_length()}'

        ax.text(self.width / 2, -0.5, title, fontsize=14, fontweight='500', ha='center', va='top')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Test the environment with direction-based actions, fixed start, and dead-end goals
    fixed_start = (0, 0)  # Top-left corner, set to None for random starts
    goal_is_deadend = True  # Goals always at dead-ends

    env = MazeEnvironment(length=8, width=8, corridor=0.5, seed=60,
                          fixed_start_node=fixed_start,
                          goal_is_deadend=goal_is_deadend)

    print(f"Environment created with {env.num_nodes} nodes")
    print(f"Action space: {env.num_actions} actions")
    print(f"  0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL")
    print(f"Fixed start node: {fixed_start if fixed_start else 'Random'}")
    print(f"Goal selection: {'Dead-ends only' if goal_is_deadend else 'All nodes'}")
    print(f"Dead-end nodes found: {len(env.deadend_nodes)}")
    print(f"Dead-end positions: {env.deadend_nodes[:5]}..." if len(env.deadend_nodes) > 5 else f"Dead-end positions: {env.deadend_nodes}")

    # Test multiple episodes
    print(f"\nTesting across 3 episodes:")
    for ep in range(3):
        state = env.reset()
        goal_degree = env.graph.degree(state['goal_pos'])
        is_deadend = state['goal_pos'] in env.deadend_nodes

        print(f"\nEpisode {ep+1}:")
        print(f"  Start: {state['current_pos']}" + (f" (always {fixed_start})" if fixed_start else " (random)"))
        print(f"  Goal: {state['goal_pos']} (degree={goal_degree}, is_deadend={is_deadend})")
        print(f"  Optimal path length: {env.get_optimal_path_length()}")

    # Take a few steps using optimal actions in one episode
    print(f"\nTaking optimal steps in one episode:")
    state = env.reset()  # Start new episode
    print(f"  Starting at: {state['current_pos']}, Goal: {state['goal_pos']}")
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDENTIFY_GOAL'}

    for i in range(100):
        optimal_action = env.get_optimal_next_action()
        if optimal_action is None:
            print("  No path to goal!")
            break

        action_name = action_names.get(optimal_action, 'UNKNOWN')
        state, reward, done, info = env.step(optimal_action, used_slow=False)
        print(f"  Step {i+1}: action={action_name} ({optimal_action}), pos={state['current_pos']}, reward={reward:.2f}, done={done}")

        if done:
            print(f"  Episode finished! Reached goal: {info['reached_goal']}")
            break

    print("\n✓ MazeEnvironment tests passed!")
