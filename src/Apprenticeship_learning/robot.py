from gridworld import GridWorld
from maxent_utils.maxent_wrapper import Trajectory
from maxent_utils.maxent import irl_causal
from maxent_utils.maxent_wrapper import ExpSga, linear_decay, Optimizer, Initializer
import numpy as np
from typing import List, Tuple
from valueiterationplanner import ValueIterationPlanner

def get_valid_neighbors(state, grid_size):
    """Get valid neighboring states (up, down, left, right)"""
    x, y = state[0], state[1]
    neighbors = []
    
    # Check all four directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        # Check if new position is within grid bounds
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            neighbors.append((new_x, new_y))
            
    return neighbors

class RobotLearner:
    def __init__(self, env: GridWorld):
        self.env = env
        self.estimated_theta = None
        self.planner = ValueIterationPlanner(env)
        
    def learn_from_demonstration(self, trajectory_data: List[Tuple[np.ndarray, str]]):
        """Implement Maximum Entropy IRL using the maxent package"""
        # Convert trajectory to maxent package format
        trajectory = Trajectory(trajectory_data)
        print(f'Robot is learning from trajectory: {trajectory.states_actions}')
        trajectory.grid_size = self.env.size
        
        p_transition = self.env.get_p_transition()
        # Setup features matrix
        features = self.env.get_feature_matrix()
        print(f'Features matrix: {features.shape}')
        
        
        # Define terminal states 
        final_state, _ = trajectory_data[-1]  # Get last (state, action) pair
        final_state_idx = int(final_state[0]) * self.env.size + int(final_state[1])
        terminal_states = [final_state_idx]

        print(f'Terminal states: {terminal_states}')
        
        # Setup optimization
        optim = ExpSga(lr=0.2).normalize_grad()
        #optim = ExpSga(lr=linear_decay(lr0=0.1))
        init = Initializer(low=-1, high=1)
        
        # Run MaxEnt IRL
        print(f'Running Causal IRL')
        self.estimated_theta = irl_causal(
            p_transition=p_transition,
            features=features,
            terminal=terminal_states,
            trajectories=[trajectory],
            optim=optim,
            init=init,
            discount=0.95,  # Add discount factor for stability
            eps=1e-4,
            eps_svf=1e-5,
            eps_lap=1e-5,
        )
        print(f'Running finished. Estimated theta: {self.estimated_theta}')
    
    def _policy(self, state: np.ndarray, theta) -> str:
        """Robot's policy based on learned rewards"""
        if theta is None:
            return np.random.choice(['N', 'S', 'E', 'W'])
            
        return self.planner.get_action(state, theta)