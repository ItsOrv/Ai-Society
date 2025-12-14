"""
Performance metrics and evaluation for multi-agent system
Includes individual, group statistics, and algorithm comparison
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import json
from datetime import datetime


class PerformanceMetrics:
    """Class for collecting and analyzing performance metrics"""
    
    def __init__(self):
        self.episode_data: List[Dict] = []
        self.step_data: List[Dict] = []
        self.agent_metrics: Dict[int, Dict] = defaultdict(lambda: {
            'total_reward': 0.0,
            'resources_collected': 0,
            'obstacles_hit': 0,
            'distance_traveled': 0.0,
            'energy_efficiency': 0.0,
            'survival_time': 0,
            'episodes': 0
        })
        self.communication_metrics: Dict = {
            'messages_sent': 0,
            'messages_received': 0,
            'cooperation_events': 0,
            'resource_sharing_events': 0
        }
        self.current_episode: Dict = {}
    
    def record_step(
        self,
        agent_id: int,
        reward: float,
        energy: float,
        position: tuple,
        action: np.ndarray,
        info: Dict
    ):
        """Record step data"""
        step_record = {
            'agent_id': agent_id,
            'reward': reward,
            'energy': energy,
            'position': position,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'timestamp': len(self.step_data),
            **info
        }
        self.step_data.append(step_record)
    
    def record_episode_end(
        self,
        agent_id: int,
        episode_length: int,
        total_reward: float,
        final_stats: Dict
    ):
        """Record episode end"""
        episode_record = {
            'agent_id': agent_id,
            'episode_length': episode_length,
            'total_reward': total_reward,
            'final_stats': final_stats,
            'timestamp': datetime.now().isoformat()
        }
        self.episode_data.append(episode_record)
        
        # Update agent metrics
        metrics = self.agent_metrics[agent_id]
        metrics['total_reward'] += total_reward
        metrics['resources_collected'] += final_stats.get('resources_collected', 0)
        metrics['obstacles_hit'] += final_stats.get('obstacles_hit', 0)
        metrics['distance_traveled'] += final_stats.get('total_distance_traveled', 0.0)
        metrics['survival_time'] += episode_length
        metrics['episodes'] += 1
        
        # Calculate energy efficiency
        if episode_length > 0:
            energy_eff = total_reward / episode_length
            metrics['energy_efficiency'] = (
                (metrics['energy_efficiency'] * (metrics['episodes'] - 1) + energy_eff) /
                metrics['episodes']
            )
    
    def record_communication_event(
        self,
        event_type: str,
        sender_id: int,
        receiver_id: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        """Record communication event"""
        if event_type == 'message_sent':
            self.communication_metrics['messages_sent'] += 1
        elif event_type == 'message_received':
            self.communication_metrics['messages_received'] += 1
        elif event_type == 'cooperation':
            self.communication_metrics['cooperation_events'] += 1
        elif event_type == 'resource_sharing':
            self.communication_metrics['resource_sharing_events'] += 1
    
    def get_agent_statistics(self, agent_id: int) -> Dict:
        """Get agent statistics"""
        if agent_id not in self.agent_metrics:
            return {}
        
        metrics = self.agent_metrics[agent_id]
        episodes = metrics['episodes']
        
        if episodes == 0:
            return {}
        
        return {
            'average_reward': metrics['total_reward'] / episodes,
            'average_resources_collected': metrics['resources_collected'] / episodes,
            'average_obstacles_hit': metrics['obstacles_hit'] / episodes,
            'average_distance_traveled': metrics['distance_traveled'] / episodes,
            'average_survival_time': metrics['survival_time'] / episodes,
            'energy_efficiency': metrics['energy_efficiency'],
            'total_episodes': episodes
        }
    
    def get_group_statistics(self) -> Dict:
        """Get group statistics"""
        if not self.agent_metrics:
            return {}
        
        all_rewards = []
        all_resources = []
        all_survival_times = []
        
        for agent_id, metrics in self.agent_metrics.items():
            if metrics['episodes'] > 0:
                all_rewards.append(metrics['total_reward'] / metrics['episodes'])
                all_resources.append(metrics['resources_collected'] / metrics['episodes'])
                all_survival_times.append(metrics['survival_time'] / metrics['episodes'])
        
        if not all_rewards:
            return {}
        
        return {
            'num_agents': len(self.agent_metrics),
            'average_reward_per_agent': np.mean(all_rewards),
            'std_reward_per_agent': np.std(all_rewards),
            'average_resources_per_agent': np.mean(all_resources),
            'average_survival_time': np.mean(all_survival_times),
            'total_episodes': sum(m['episodes'] for m in self.agent_metrics.values()),
            'communication_stats': self.communication_metrics.copy()
        }
    
    def get_learning_curve(self, agent_id: int, window_size: int = 100) -> Dict[str, List[float]]:
        """Compute learning curve"""
        agent_episodes = [
            ep for ep in self.episode_data
            if ep['agent_id'] == agent_id
        ]
        
        if not agent_episodes:
            return {'rewards': [], 'lengths': []}
        
        rewards = [ep['total_reward'] for ep in agent_episodes]
        lengths = [ep['episode_length'] for ep in agent_episodes]
        
        # Moving average
        if len(rewards) > window_size:
            rewards_ma = []
            lengths_ma = []
            for i in range(window_size, len(rewards)):
                rewards_ma.append(np.mean(rewards[i-window_size:i]))
                lengths_ma.append(np.mean(lengths[i-window_size:i]))
            return {'rewards': rewards_ma, 'lengths': lengths_ma}
        
        return {'rewards': rewards, 'lengths': lengths}
    
    def compare_agents(self) -> Dict:
        """Compare agent performance"""
        comparisons = {}
        
        for agent_id in self.agent_metrics.keys():
            stats = self.get_agent_statistics(agent_id)
            if stats:
                comparisons[agent_id] = {
                    'rank_by_reward': 0,
                    'rank_by_efficiency': 0,
                    'rank_by_survival': 0,
                    **stats
                }
        
        # Ranking
        if comparisons:
            sorted_by_reward = sorted(
                comparisons.items(),
                key=lambda x: x[1]['average_reward'],
                reverse=True
            )
            sorted_by_efficiency = sorted(
                comparisons.items(),
                key=lambda x: x[1]['energy_efficiency'],
                reverse=True
            )
            sorted_by_survival = sorted(
                comparisons.items(),
                key=lambda x: x[1]['average_survival_time'],
                reverse=True
            )
            
            for rank, (agent_id, _) in enumerate(sorted_by_reward, 1):
                comparisons[agent_id]['rank_by_reward'] = rank
            for rank, (agent_id, _) in enumerate(sorted_by_efficiency, 1):
                comparisons[agent_id]['rank_by_efficiency'] = rank
            for rank, (agent_id, _) in enumerate(sorted_by_survival, 1):
                comparisons[agent_id]['rank_by_survival'] = rank
        
        return comparisons
    
    def export_to_json(self, filepath: str):
        """Export to JSON"""
        try:
            export_data = {
                'agent_metrics': {
                    str(k): v for k, v in self.agent_metrics.items()
                },
                'communication_metrics': self.communication_metrics,
                'group_statistics': self.get_group_statistics(),
                'agent_comparisons': self.compare_agents(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to write metrics to {filepath}: {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize metrics data: {e}")
    
    def reset(self):
        """Reset all metrics"""
        self.episode_data.clear()
        self.step_data.clear()
        self.agent_metrics.clear()
        self.communication_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'cooperation_events': 0,
            'resource_sharing_events': 0
        }
        self.current_episode.clear()


class AlgorithmComparator:
    """Compare different RL algorithms"""
    
    def __init__(self):
        self.algorithm_results: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_result(
        self,
        algorithm_name: str,
        episode_reward: float,
        episode_length: int,
        additional_metrics: Optional[Dict] = None
    ):
        """Add algorithm result"""
        result = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            **(additional_metrics or {})
        }
        self.algorithm_results[algorithm_name].append(result)
    
    def compare_algorithms(self) -> Dict:
        """Compare algorithms"""
        comparison = {}
        
        for algo_name, results in self.algorithm_results.items():
            if not results:
                continue
            
            rewards = [r['episode_reward'] for r in results]
            lengths = [r['episode_length'] for r in results]
            
            comparison[algo_name] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'num_episodes': len(results),
                'best_reward': np.max(rewards),
                'worst_reward': np.min(rewards)
            }
        
        return comparison
    
    def get_best_algorithm(self) -> Optional[str]:
        """Find best algorithm"""
        comparison = self.compare_algorithms()
        if not comparison:
            return None
        
        best_algo = max(
            comparison.items(),
            key=lambda x: x[1]['mean_reward']
        )
        return best_algo[0]


