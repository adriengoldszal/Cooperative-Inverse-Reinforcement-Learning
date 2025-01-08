import numpy as np
import matplotlib.pyplot as plt
from run_utils import create_timestamped_dir, visualize_policies, plot_comparison_metrics
import os
import json

from gridworld import GridWorld
from human import HumanTeacher
from robot import RobotLearner
from maxent_utils.maxent import compute_trajectory_kl_divergence, initial_probabilities_from_trajectories
from maxent_utils.maxent_wrapper import Trajectory


def run_experiment(results_path, n_trials: int = 2, grid_size: int = 10, horizon: int = 16):
    """Run the complete experiment"""
    results = {
        'expert_3': {'l2_norms': [], 'regret': [], 'kl_divergence': []},
        'best_response_3': {'l2_norms': [], 'regret': [], 'kl_divergence': []},
        'expert_10': {'l2_norms': [], 'regret': [], 'kl_divergence': []},
        'best_response_10': {'l2_norms': [], 'regret': [], 'kl_divergence': []}
    }
    
    
    for trial in range(n_trials):
        print(f"Trial: {trial}")
        for n_features in [3, 10]:
            # Create environment and true reward parameters
            env = GridWorld(size=grid_size, n_features=n_features)
            print('feature centers:', env.feature_centers)
            
            #We use a uniform distribution in [-1, 1] for the prior on theta 
            true_theta = np.random.uniform(-1, 1, n_features)
            
            # Test both policies
            for policy_type in ['expert', 'best_response']:
                # Create agents
                human = HumanTeacher(env, true_theta, policy_type)
                robot = RobotLearner(env)
                
                trajectory = human.demonstrate(horizon // 2)
                
                # Learning phase
                robot.learn_from_demonstration(trajectory)
                
                if policy_type == 'expert':
                    expert_trajectory, robot_expert_theta = trajectory, robot.estimated_theta
                    print("Expert trajectory:", expert_trajectory)
                else:
                    br_trajectory, robot_br_theta = trajectory, robot.estimated_theta
                    print("BR trajectory:", br_trajectory)
                    print(f"Current trial iteration: {trial}")
                    visualize_policies(env, true_theta, expert_trajectory, robot_expert_theta, br_trajectory, robot_br_theta, results_dir, trial)
                
                # Deployment phase
                initial_state = env.reset(random_state=True)
                reward_estimated_theta = 0
                reward_true_theta = 0
                
                state = initial_state.copy()
                for _ in range(horizon // 2):
                    action = robot._policy(state, robot.estimated_theta)
                    reward_estimated_theta += env.get_reward(state, robot.estimated_theta)
                    state = env.step(action)
                
                state = initial_state.copy()
                for _ in range(horizon // 2):
                    action = robot._policy(state, true_theta)
                    reward_true_theta += env.get_reward(state, true_theta)
                    state = env.step(action)
                    
                # Calculate measurements
                l2 = np.linalg.norm(true_theta - robot.estimated_theta)
                regret = np.abs(reward_true_theta - reward_estimated_theta)
                
                p_transition = env.get_p_transition()
                features = env.get_feature_matrix()
                n_states = grid_size * grid_size
                                
                # Define terminal states 
                trajectory_obj = Trajectory(trajectory)
                trajectory_obj.grid_size = env.size
                final_state_idx = trajectory_obj.states()[-1]  # Get the last state index
                terminal_states = [final_state_idx]
                    
                theta_hat = robot.estimated_theta
                theta_gt = true_theta
                kl_trajectory = Trajectory(trajectory)
                kl_trajectory.grid_size = env.size
                p_initial = initial_probabilities_from_trajectories(n_states, [kl_trajectory])
                print(f'total initial probabilities: {np.sum(p_initial)}')
                
                kl = compute_trajectory_kl_divergence(p_transition, features, theta_hat, theta_gt, terminal_states, 
                                   p_initial)
                
                # Store results
                key = f"{policy_type}_{n_features}"
                results[key]['l2_norms'].append(l2)
                results[key]['regret'].append(regret)
                results[key]['kl_divergence'].append(kl)
                
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
                
                
    
    # Compute averages for each policy and feature count
    averages = {}
    for key in results:
        averages[key] = {
            'avg_l2': np.mean(results[key]['l2_norms']),
            'std_l2': np.std(results[key]['l2_norms']),
            'avg_regret': np.mean(results[key]['regret'], axis=0),
            'std_regret': np.std(results[key]['regret'], axis=0),
            'avg_kl': np.mean(results[key]['kl_divergence'], axis=0),
            'std_kl': np.std(results[key]['kl_divergence'], axis=0)
        }
    
    return results, averages

# Example usage:
if __name__ == "__main__":
    
    results_dir = create_timestamped_dir()
    print(f"Results will be saved in: {results_dir}")
    results_path = os.path.join(results_dir, "experiment_results.json")
    results, averages = run_experiment(results_path=results_path)
    
    fig  = plot_comparison_metrics(results, save_path=results_dir)
    
    

    # Plotting the distributions
    plt.figure(figsize=(10, 6))

    for n_features in [3, 10]:
        # Get the results for the given number of features
        br_l2 = results[f'best_response_{n_features}']['l2_norms']
        expert_l2 = results[f'expert_{n_features}']['l2_norms']
        
        br_regret = results[f'best_response_{n_features}']['regret']
        expert_regret = results[f'expert_{n_features}']['regret']
        
        br_kl = results[f'best_response_{n_features}']['kl_divergence']
        expert_kl = results[f'expert_{n_features}']['kl_divergence']

        
        # Plot L2 norms for the feature count
        plt.hist(br_l2, alpha=0.5, label=f"BR (L2) - {n_features} features (avg={averages[f'best_response_{n_features}']['avg_l2']:.3f})")
        plt.hist(expert_l2, alpha=0.5, label=f"Expert (L2) - {n_features} features (avg={averages[f'expert_{n_features}']['avg_l2']:.3f})")
        
        # Plot regret for the feature count
        plt.hist(br_regret, alpha=0.5, linestyle='dashed', label=f"BR (Regret) - {n_features} features (avg={averages[f'best_response_{n_features}']['avg_regret']:.3f})")
        plt.hist(expert_regret, alpha=0.5, linestyle='dashed', label=f"Expert (Regret) - {n_features} features (avg={averages[f'expert_{n_features}']['avg_regret']:.3f})")

        plt.hist(br_kl, alpha=0.5, linestyle='dotted', label=f"BR (KL) - {n_features} features (avg={averages[f'best_response_{n_features}']['avg_kl']:.3f})")
        plt.hist(expert_kl, alpha=0.5, linestyle='dotted', label=f"Expert (KL) - {n_features} features (avg={averages[f'expert_{n_features}']['avg_kl']:.3f})")
    
    # Customize the plot
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of L2 Norms, KL divergence and Regret')
    plt.show()
    
