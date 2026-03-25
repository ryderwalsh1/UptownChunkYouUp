"""
Maze Environment Wrapper

Wraps MazeGraph from corridors.py into a standard RL environment interface.
Handles state/goal representation and reward structure.
"""

import numpy as np
import networkx as nx
from corridors import MazeGraph


class MazeEnvironment:
    def __init__(self, length=8, width=8, corridor=0.5, seed=None, control_cost=0.01):
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
        """
        self.length = length
        self.width = width
        self.corridor = corridor
        self.seed = seed
        self.control_cost = control_cost

        # Generate maze graph
        self.maze = MazeGraph(length=length, width=width, corridor=corridor, seed=seed)
        self.graph = self.maze.get_graph()

        # Create node mappings
        self.nodes_list = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        self.num_nodes = len(self.nodes_list)
        self.num_actions = self.num_nodes  # Actions are node indices (allows teleportation)

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
            Starting position (row, col). If None, random.
        goal_pos : tuple, optional
            Goal position (row, col). If None, random (different from start).

        Returns:
        --------
        state : dict
            Initial state with current_pos, goal_pos, and step_count
        """
        # Set start position
        if start_pos is None:
            self.current_pos = self.nodes_list[np.random.randint(self.num_nodes)]
        else:
            self.current_pos = start_pos

        # Set goal position
        if goal_pos is None:
            # Choose random goal different from start
            possible_goals = [n for n in self.nodes_list if n != self.current_pos]
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
            Action (node index to move to - allows teleportation)
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
        # Action is directly the node index to move to
        # Networks can "teleport" to any node
        if 0 <= action < self.num_nodes:
            self.current_pos = self.idx_to_node[action]
            invalid_move = False
        else:
            # Invalid action index - stay in place
            invalid_move = True

        self.step_count += 1

        # Calculate reward
        reward = 0.0
        done = False

        # Goal reward
        if self.current_pos == self.goal_pos:
            reward += 10.0  # Large positive reward for reaching goal
            done = True

        # Step penalty (efficiency term - encourages shorter paths)
        reward -= 0.1

        # Invalid move penalty
        if invalid_move:
            reward -= 1.0

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
            'reached_goal': self.current_pos == self.goal_pos
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
        Get optimal next action (node index) from pos toward goal using shortest path.

        Returns None if no path exists or already at goal.
        """
        if pos is None:
            pos = self.current_pos
        if goal is None:
            goal = self.goal_pos

        if pos == goal:
            return None

        if not nx.has_path(self.graph, pos, goal):
            return None

        path = nx.shortest_path(self.graph, pos, goal)
        if len(path) < 2:
            return None

        next_pos = path[1]

        # Action is directly the node index
        return self.node_to_idx[next_pos]

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
    # Test the environment
    env = MazeEnvironment(length=8, width=8, corridor=0.5, seed=60)

    print(f"Environment created with {env.num_nodes} nodes")
    print(f"Action space: {env.num_actions} actions (up, down, left, right)")

    # Test episode
    state = env.reset()
    print(f"\nEpisode started:")
    print(f"  Start: {state['current_pos']}")
    print(f"  Goal: {state['goal_pos']}")
    print(f"  Optimal path length: {env.get_optimal_path_length()}")

    # Take a few steps using optimal actions
    print(f"\nTaking optimal steps:")
    for i in range(100):
        if env.current_pos == env.goal_pos:
            print("  Reached goal!")
            break

        optimal_action = env.get_optimal_next_action()
        if optimal_action is None:
            print("  No path to goal!")
            break

        state, reward, done, info = env.step(optimal_action, used_slow=False)
        print(f"  Step {i+1}: action={optimal_action}, pos={state['current_pos']}, reward={reward:.2f}, done={done}")

        if done:
            print(f"  Episode finished! Reached goal: {info['reached_goal']}")
            break
