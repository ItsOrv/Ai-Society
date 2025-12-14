"""
Main entry point for multi-agent reinforcement learning simulation
"""

import numpy as np
import argparse
import time
from typing import List, Dict
import json

from environment import Advanced3DEnvironment
from robot import IntelligentRobot
from communication import CommunicationNetwork
from metrics import PerformanceMetrics, AlgorithmComparator
from visualization import SimulationVisualizer, MetricsPlotter


def create_robots(
    num_robots: int,
    base_environment: Advanced3DEnvironment,
    communication_network: CommunicationNetwork,
    cooperation_modes: List[str] = None
) -> List[IntelligentRobot]:
    """Create robot instances, each with its own environment"""
    if num_robots < 1:
        raise ValueError(f"num_robots must be at least 1, got {num_robots}")
    if cooperation_modes is None:
        cooperation_modes = ['mixed'] * num_robots
    if len(cooperation_modes) < num_robots:
        cooperation_modes.extend(['mixed'] * (num_robots - len(cooperation_modes)))
    
    robots = []
    grid_size = base_environment.grid_size
    
    for i in range(num_robots):
        env = Advanced3DEnvironment(
            grid_size=grid_size,
            num_resources=base_environment.num_resources,
            num_obstacles=base_environment.num_obstacles,
            resource_respawn_rate=base_environment.resource_respawn_rate,
            obstacle_movement=base_environment.obstacle_movement,
            energy_decay_rate=base_environment.energy_decay_rate,
            resource_energy_value=base_environment.resource_energy_value,
            collision_penalty=base_environment.collision_penalty,
            max_steps=base_environment.max_steps,
            seed=base_environment.seed
        )
        
        # Find valid starting position
        attempts = 0
        while attempts < 1000:
            pos = tuple(np.random.randint(0, grid_size, size=3))
            if env.grid[pos] == 0:
                break
            attempts += 1
        if attempts >= 1000:
            raise RuntimeError(f"Failed to find valid starting position for robot {i} after 1000 attempts")
        
        robot = IntelligentRobot(
            robot_id=i,
            initial_position=pos,
            environment=env,
            communication_network=communication_network,
            obs_dim=41,
            action_dim=3,
            cooperation_mode=cooperation_modes[i] if i < len(cooperation_modes) else 'mixed',
            exploration_rate=0.1,
            hidden_sizes=[128, 128, 64]
        )
        robots.append(robot)
    
    return robots


def train_episode(
    robots: List[IntelligentRobot],
    base_environment: Advanced3DEnvironment,
    communication_network: CommunicationNetwork,
    max_steps: int = 500,
    render: bool = False,
    visualizer: SimulationVisualizer = None,
    episode_num: int = 0
) -> Dict:
    """Run a single training episode"""
    base_obs, _ = base_environment.reset()
    
    # Reset all robots
    for robot in robots:
        robot.environment.reset()
        attempts = 0
        while attempts < 1000:
            pos = tuple(np.random.randint(0, robot.environment.grid_size, size=3))
            if robot.environment.grid[pos] == 0:
                robot.reset(pos)
                break
            attempts += 1
        if attempts >= 1000:
            raise RuntimeError(f"Failed to find valid starting position for robot {robot.robot_id} after 1000 attempts")
    
    episode_data = {
        'rewards': {r.robot_id: 0.0 for r in robots},
        'lengths': {r.robot_id: 0 for r in robots},
        'resources': {r.robot_id: 0 for r in robots},
        'final_stats': {}
    }
    
    step = 0
    done_robots = set()
    
    while step < max_steps and len(done_robots) < len(robots):
        communication_network.update_timestamp(step)
        
        robots_data = []
        for robot in robots:
            if robot.robot_id in done_robots:
                continue
            
            step_result = robot.step()
            
            episode_data['rewards'][robot.robot_id] += step_result['reward']
            episode_data['lengths'][robot.robot_id] += 1
            
            if step_result.get('info', {}).get('resource_collected', False):
                episode_data['resources'][robot.robot_id] += 1
            
            if step_result['done'] or step_result['truncated']:
                done_robots.add(robot.robot_id)
            
            robots_data.append({
                'id': robot.robot_id,
                'position': step_result['position'],
                'path': robot.path,
                'energy': robot.energy
            })
        
        if render and visualizer and step % 5 == 0:
            display_env = robots[0].environment if robots else base_environment
            visualizer.update_visualization(
                robots_data,
                display_env.resource_positions,
                display_env.obstacle_positions,
                episode=episode_num,
                step=step
            )
        
        step += 1
    
    # Update policies
    for robot in robots:
        last_obs = robot.get_enhanced_observation()
        robot.update_policy(last_obs, robot.robot_id in done_robots)
        episode_data['final_stats'][robot.robot_id] = robot.get_stats()
    
    return episode_data


def main():
    parser = argparse.ArgumentParser(description='Multi-Agent RL Simulation')
    parser.add_argument('--num_robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--grid_size', type=int, default=15, help='Grid size')
    parser.add_argument('--render', action='store_true', help='Render simulation')
    parser.add_argument('--save_metrics', type=str, default=None, help='Path to save metrics')
    parser.add_argument('--cooperation_mode', type=str, default='mixed', 
                       choices=['competitive', 'cooperative', 'mixed'],
                       help='Cooperation mode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        import random
        random.seed(args.seed)
    
    print("=" * 60)
    print("Multi-Agent Reinforcement Learning Simulation")
    print("=" * 60)
    print(f"Number of robots: {args.num_robots}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Grid size: {args.grid_size}")
    print(f"Cooperation mode: {args.cooperation_mode}")
    print("=" * 60)
    
    environment = Advanced3DEnvironment(
        grid_size=args.grid_size,
        num_resources=8,
        num_obstacles=15,
        resource_respawn_rate=0.05,
        max_steps=500,
        seed=args.seed
    )
    
    communication_network = CommunicationNetwork()
    
    cooperation_modes = [args.cooperation_mode] * args.num_robots
    robots = create_robots(
        args.num_robots,
        environment,
        communication_network,
        cooperation_modes
    )
    
    metrics = PerformanceMetrics()
    
    visualizer = None
    if args.render:
        visualizer = SimulationVisualizer(args.grid_size, args.num_robots)
        visualizer.initialize_plot()
    
    print("\nStarting training...")
    episode_rewards = []
    episode_lengths = []
    
    start_time = time.time()
    
    for episode in range(args.num_episodes):
        episode_data = train_episode(
            robots,
            environment,
            communication_network,
            max_steps=500,
            render=args.render,
            visualizer=visualizer,
            episode_num=episode
        )
        
        for robot in robots:
            robot_stats = episode_data['final_stats'][robot.robot_id]
            metrics.record_episode_end(
                robot.robot_id,
                episode_data['lengths'][robot.robot_id],
                episode_data['rewards'][robot.robot_id],
                robot_stats
            )
        
        avg_reward = np.mean(list(episode_data['rewards'].values()))
        avg_length = np.mean(list(episode_data['lengths'].values()))
        episode_rewards.append(avg_reward)
        episode_lengths.append(avg_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{args.num_episodes}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Length: {avg_length:.1f}")
            print(f"  Total Resources Collected: {sum(episode_data['resources'].values())}")
            
            for robot in robots:
                stats = metrics.get_agent_statistics(robot.robot_id)
                if stats:
                    print(f"  Robot {robot.robot_id}: Reward={stats['average_reward']:.2f}, "
                          f"Resources={stats['average_resources_collected']:.1f}")
        
        if args.render and visualizer:
            learning_curves = {}
            for robot in robots:
                curve = metrics.get_learning_curve(robot.robot_id, window_size=20)
                learning_curves[robot.robot_id] = curve
            
            visualizer.plot_statistics(episode_rewards, episode_lengths, learning_curves.get(0))
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    
    group_stats = metrics.get_group_statistics()
    print(f"Group Statistics:")
    print(f"  Average Reward per Agent: {group_stats.get('average_reward_per_agent', 0):.2f}")
    print(f"  Average Resources per Agent: {group_stats.get('average_resources_per_agent', 0):.2f}")
    print(f"  Average Survival Time: {group_stats.get('average_survival_time', 0):.1f}")
    
    print(f"\nCommunication Statistics:")
    comm_stats = group_stats.get('communication_stats', {})
    print(f"  Messages Sent: {comm_stats.get('messages_sent', 0)}")
    print(f"  Messages Received: {comm_stats.get('messages_received', 0)}")
    print(f"  Cooperation Events: {comm_stats.get('cooperation_events', 0)}")
    
    print(f"\nIndividual Agent Statistics:")
    for robot in robots:
        stats = metrics.get_agent_statistics(robot.robot_id)
        if stats:
            print(f"  Robot {robot.robot_id}:")
            print(f"    Average Reward: {stats['average_reward']:.2f}")
            print(f"    Resources Collected: {stats['average_resources_collected']:.1f}")
            print(f"    Survival Time: {stats['average_survival_time']:.1f}")
            print(f"    Energy Efficiency: {stats['energy_efficiency']:.4f}")
    
    if args.save_metrics:
        metrics.export_to_json(args.save_metrics)
        print(f"\nMetrics saved to {args.save_metrics}")
    
    if args.render and visualizer:
        print("\nGenerating final plots...")
        
        agent_stats = {}
        for robot in robots:
            agent_stats[robot.robot_id] = metrics.get_agent_statistics(robot.robot_id)
        
        MetricsPlotter.plot_agent_comparison(agent_stats)
        
        learning_curves = {}
        for robot in robots:
            learning_curves[robot.robot_id] = metrics.get_learning_curve(robot.robot_id)
        
        MetricsPlotter.plot_learning_curves(learning_curves)
        
        print("Plots generated. Close the windows to exit.")
        input("Press Enter to exit...")
        visualizer.close()
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
