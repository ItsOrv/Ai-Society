"""
Simple Text-Based Visualization System
No external plotting libraries - pure text output for statistics and state display
"""

import numpy as np
from typing import List, Dict, Optional


class SimulationVisualizer:
    """
    Simple text-based visualizer - no matplotlib dependency
    Outputs statistics and state information to console
    """
    
    def __init__(self, grid_size: int, num_robots: int):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.learning_curves: Dict[int, List[float]] = {}
        
    def initialize_plot(self):
        """Initialize visualization (text-based)"""
        print("=" * 80)
        print("Multi-Agent RL Simulation - Text-Based Visualization")
        print("=" * 80)
    
    def update_visualization(
        self,
        robots_data: List[Dict],
        resource_positions: List[tuple],
        obstacle_positions: List[tuple],
        episode: int = 0,
        step: int = 0
    ):
        """Update and display current state"""
        if step % 50 == 0:  # Print every 50 steps
            print(f"\nEpisode {episode}, Step {step}")
            print("-" * 80)
            for robot_data in robots_data:
                pos = robot_data['position']
                energy = robot_data.get('energy', 0)
                print(f"Robot {robot_data['id']}: Position={pos}, Energy={energy:.1f}")
            print(f"Resources: {len(resource_positions)}, Obstacles: {len(obstacle_positions)}")
    
    def plot_statistics(
        self,
        episode_rewards: List[float],
        episode_lengths: List[int],
        learning_curve: Optional[Dict] = None
    ):
        """Display statistics as text"""
        self.episode_rewards = episode_rewards
        self.episode_lengths = episode_lengths
        
        if len(episode_rewards) > 0 and len(episode_rewards) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            
            print(f"\nRecent Performance (last 10 episodes):")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Length: {avg_length:.1f}")
    
    def plot_communication_stats(
        self,
        messages_sent: List[int],
        messages_received: List[int]
    ):
        """Display communication statistics"""
        if len(messages_sent) > 0:
            total_sent = sum(messages_sent)
            total_received = sum(messages_received)
            print(f"\nCommunication Stats:")
            print(f"  Total Messages Sent: {total_sent}")
            print(f"  Total Messages Received: {total_received}")
    
    def save_figure(self, filepath: str):
        """Save statistics to text file"""
        with open(filepath, 'w') as f:
            f.write("Simulation Statistics\n")
            f.write("=" * 80 + "\n")
            if self.episode_rewards:
                f.write(f"Total Episodes: {len(self.episode_rewards)}\n")
                f.write(f"Average Reward: {np.mean(self.episode_rewards):.2f}\n")
                f.write(f"Max Reward: {np.max(self.episode_rewards):.2f}\n")
                f.write(f"Min Reward: {np.min(self.episode_rewards):.2f}\n")
    
    def close(self):
        """Close visualizer"""
        print("\n" + "=" * 80)
        print("Simulation Complete")
        print("=" * 80)


class MetricsPlotter:
    """Text-based metrics plotter"""
    
    @staticmethod
    def plot_agent_comparison(metrics_data: Dict, save_path: Optional[str] = None):
        """Display agent comparison as text"""
        print("\n" + "=" * 80)
        print("Agent Comparison")
        print("=" * 80)
        
        for agent_id, stats in sorted(metrics_data.items()):
            if stats:
                print(f"\nAgent {agent_id}:")
                print(f"  Average Reward: {stats.get('average_reward', 0):.2f}")
                print(f"  Resources Collected: {stats.get('average_resources_collected', 0):.1f}")
                print(f"  Survival Time: {stats.get('average_survival_time', 0):.1f}")
                print(f"  Energy Efficiency: {stats.get('energy_efficiency', 0):.4f}")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Agent Comparison\n")
                f.write("=" * 80 + "\n")
                for agent_id, stats in sorted(metrics_data.items()):
                    if stats:
                        f.write(f"\nAgent {agent_id}:\n")
                        for key, value in stats.items():
                            f.write(f"  {key}: {value}\n")
    
    @staticmethod
    def plot_learning_curves(learning_curves: Dict[str, Dict], save_path: Optional[str] = None):
        """Display learning curves as text"""
        print("\n" + "=" * 80)
        print("Learning Curves")
        print("=" * 80)
        
        for agent_id, curve_data in sorted(learning_curves.items()):
            rewards = curve_data.get('rewards', [])
            if rewards:
                recent = rewards[-20:] if len(rewards) > 20 else rewards
                avg = np.mean(recent)
                print(f"Agent {agent_id}: Recent Average Reward = {avg:.2f}")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Learning Curves\n")
                f.write("=" * 80 + "\n")
                for agent_id, curve_data in sorted(learning_curves.items()):
                    rewards = curve_data.get('rewards', [])
                    if rewards:
                        f.write(f"\nAgent {agent_id}:\n")
                        f.write(f"  Total Episodes: {len(rewards)}\n")
                        f.write(f"  Final Average: {np.mean(rewards[-20:]):.2f}\n")
