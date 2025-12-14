"""
Robot class with learning, communication, and cooperation capabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ppo_agent import PPOAgent
from communication import CommunicationProtocol, CommunicationNetwork
from environment import Advanced3DEnvironment


class IntelligentRobot:
    """Intelligent robot with advanced capabilities"""
    
    def __init__(
        self,
        robot_id: int,
        initial_position: Tuple[int, int, int],
        environment: Advanced3DEnvironment,
        communication_network: CommunicationNetwork,
        obs_dim: int = 32,
        action_dim: int = 3,
        cooperation_mode: str = 'competitive',
        exploration_rate: float = 0.1,
        hidden_sizes: List[int] = [128, 128, 64]
    ):
        self.robot_id = robot_id
        self.initial_position = initial_position
        self.environment = environment
        self.communication_network = communication_network
        self.cooperation_mode = cooperation_mode
        self.exploration_rate = exploration_rate
        
        self.agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            gamma=0.99,
            clip_epsilon=0.2,
            update_epochs=10,
            batch_size=64
        )
        
        self.communication = CommunicationProtocol(robot_id, communication_network)
        
        self.current_position = np.array(initial_position, dtype=np.float32)
        self.energy = 100.0
        self.path: List[Tuple[int, int, int]] = [tuple(initial_position)]
        self.episode_reward = 0.0
        self.episode_length = 0
        
        self.visited_positions: set = set()
        self.resource_memory: Dict[Tuple[int, int, int], int] = {}
        self.obstacle_memory: set = set()
        self.last_action = np.zeros(3)
        
        self.stats = {
            'resources_collected': 0,
            'obstacles_hit': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'cooperation_actions': 0,
            'total_distance': 0.0
        }
    
    def reset(self, new_position: Optional[Tuple[int, int, int]] = None):
        """Reset robot state"""
        if new_position is None:
            new_position = self.initial_position
        
        self.current_position = np.array(new_position, dtype=np.float32)
        self.energy = 100.0
        self.path = [tuple(new_position)]
        self.episode_reward = 0.0
        self.episode_length = 0
        self.visited_positions.clear()
        self.resource_memory.clear()
        self.obstacle_memory.clear()
        self.last_action = np.zeros(3)
        
        self.stats['resources_collected'] = 0
        self.stats['obstacles_hit'] = 0
        self.stats['messages_sent'] = 0
        self.stats['messages_received'] = 0
        self.stats['cooperation_actions'] = 0
        self.stats['total_distance'] = 0.0
    
    def get_enhanced_observation(self) -> np.ndarray:
        """Build enhanced observation with communication info"""
        base_obs = self.environment._get_observation()
        
        comm_features = self.communication.get_communication_features(
            tuple(self.current_position.astype(int))
        )
        
        memory_features = np.array([
            len(self.visited_positions) / 100.0,
            len(self.resource_memory) / 10.0,
            len(self.obstacle_memory) / 10.0
        ], dtype=np.float32)
        
        energy_features = np.array([
            self.energy / 100.0,
            self.episode_length / 500.0
        ], dtype=np.float32)
        
        enhanced_obs = np.concatenate([
            base_obs,
            comm_features,
            memory_features,
            energy_features
        ])
        
        return enhanced_obs.astype(np.float32)
    
    def select_action(self, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action considering communication"""
        obs = self.get_enhanced_observation()
        
        if not deterministic and np.random.random() < self.exploration_rate:
            action = np.random.uniform(-1, 1, size=3)
            log_prob = -np.log(2.0) * 3
            value = self.agent.value_net.forward(obs.reshape(1, -1), training=False)[0]
        else:
            action, log_prob, value = self.agent.select_action(obs, deterministic)
        
        action = self._modify_action_with_communication(action, obs)
        
        return action, log_prob, value
    
    def _modify_action_with_communication(
        self,
        action: np.ndarray,
        obs: np.ndarray
    ) -> np.ndarray:
        """Modify action based on received messages"""
        processed_messages = self.communication.process_messages(
            tuple(self.current_position.astype(int))
        )
        
        if processed_messages['new_resources']:
            nearest_resource = self.communication.get_nearest_known_resource(
                tuple(self.current_position.astype(int))
            )
            if nearest_resource:
                direction = np.array(nearest_resource) - self.current_position
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = 0.6 * action + 0.4 * direction
        
        if processed_messages['new_obstacles']:
            for obstacle_pos in processed_messages['new_obstacles']:
                direction_away = self.current_position - np.array(obstacle_pos)
                distance = np.linalg.norm(direction_away)
                if distance < 5.0:
                    direction_away = direction_away / (distance + 1e-8)
                    action = 0.7 * action + 0.3 * direction_away
        
        return np.clip(action, -1.0, 1.0)
    
    def step(self) -> Dict:
        """Execute one step"""
        action, log_prob, value = self.select_action()
        self.last_action = action.copy()
        
        obs, reward, done, truncated, info = self.environment.step(action)
        
        self.current_position = self.environment.agent_pos.copy()
        self.energy = self.environment.agent_energy
        self.episode_reward += reward
        self.episode_length += 1
        
        current_pos_int = tuple(self.current_position.astype(int))
        if current_pos_int not in self.path or len(self.path) == 0:
            self.path.append(current_pos_int)
        self.visited_positions.add(current_pos_int)
        
        if info.get('resource_collected', False):
            self.stats['resources_collected'] += 1
            self.resource_memory[current_pos_int] = self.communication_network.current_timestamp
            if self.cooperation_mode in ['cooperative', 'mixed']:
                self.communication.send_resource_depleted(current_pos_int)
                self.stats['messages_sent'] += 1
        
        if info.get('obstacle_hit', False):
            self.stats['obstacles_hit'] += 1
            self.obstacle_memory.add(current_pos_int)
            self.communication.send_obstacle_warning(current_pos_int)
            self.stats['messages_sent'] += 1
        
        resources_nearby = self._sense_resources()
        if resources_nearby:
            resource_pos = resources_nearby[0]
            if self.cooperation_mode in ['cooperative', 'mixed']:
                self.communication.send_resource_found(
                    resource_pos,
                    self.environment.resource_energy_value
                )
                self.stats['messages_sent'] += 1
        
        processed = self.communication.process_messages(current_pos_int)
        self.stats['messages_received'] += len(processed.get('new_resources', []))
        self.stats['messages_received'] += len(processed.get('new_obstacles', []))
        
        enhanced_obs = self.get_enhanced_observation()
        self.agent.store_transition(
            enhanced_obs,
            action,
            reward,
            value,
            log_prob,
            done or truncated
        )
        
        if len(self.path) > 1:
            distance = np.linalg.norm(
                np.array(self.path[-1]) - np.array(self.path[-2])
            )
            self.stats['total_distance'] += distance
        
        return {
            'observation': enhanced_obs,
            'action': action,
            'reward': reward,
            'done': done,
            'truncated': truncated,
            'info': info,
            'energy': self.energy,
            'position': current_pos_int
        }
    
    def _sense_resources(self, radius: int = 2) -> List[Tuple[int, int, int]]:
        """Detect resources in nearby area"""
        resources = []
        pos_int = tuple(self.current_position.astype(int))
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    check_pos = (
                        pos_int[0] + dx,
                        pos_int[1] + dy,
                        pos_int[2] + dz
                    )
                    if (0 <= check_pos[0] < self.environment.grid_size and
                        0 <= check_pos[1] < self.environment.grid_size and
                        0 <= check_pos[2] < self.environment.grid_size):
                        if check_pos in self.environment.resource_positions:
                            resources.append(check_pos)
        
        return resources
    
    def update_policy(self, last_obs: Optional[np.ndarray] = None, last_done: bool = False):
        """Update policy"""
        if last_obs is None:
            last_obs = self.get_enhanced_observation()
        self.agent.update(last_obs, last_done)
    
    def get_stats(self) -> Dict:
        """Get robot statistics"""
        return {
            'robot_id': self.robot_id,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'energy': self.energy,
            'position': tuple(self.current_position.astype(int)),
            'resources_collected': self.stats['resources_collected'],
            'obstacles_hit': self.stats['obstacles_hit'],
            'messages_sent': self.stats['messages_sent'],
            'messages_received': self.stats['messages_received'],
            'total_distance': self.stats['total_distance'],
            'path_length': len(self.path),
            'agent_stats': self.agent.get_stats()
        }
    
    def should_cooperate_with(self, other_robot_id: int) -> bool:
        """Decide whether to cooperate with another robot"""
        if self.cooperation_mode == 'competitive':
            return False
        elif self.cooperation_mode == 'cooperative':
            return True
        else:
            return self.communication.should_cooperate(other_robot_id)
