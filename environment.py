"""
3D environment for multi-agent simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ActionSpace:
    """Action space definition"""
    
    def __init__(self, low: float = -1.0, high: float = 1.0, shape: Tuple[int, ...] = (3,)):
        self.low = np.array([low] * shape[0], dtype=np.float32)
        self.high = np.array([high] * shape[0], dtype=np.float32)
        self.shape = shape
    
    def sample(self) -> np.ndarray:
        """Sample random action from uniform distribution"""
        return np.random.uniform(self.low, self.high, size=self.shape)
    
    def contains(self, action: np.ndarray) -> bool:
        """Check if action is within bounds"""
        return np.all(action >= self.low) and np.all(action <= self.high)


class ObservationSpace:
    """Observation space definition"""
    
    def __init__(self, low: float, high: float, shape: Tuple[int, ...]):
        self.low = np.array([low] * shape[0], dtype=np.float32)
        self.high = np.array([high] * shape[0], dtype=np.float32)
        self.shape = shape
    
    def contains(self, observation: np.ndarray) -> bool:
        """Check if observation is within bounds"""
        return np.all(observation >= self.low) and np.all(observation <= self.high)


class Advanced3DEnvironment:
    """3D grid-based environment for multi-agent simulation"""
    
    def __init__(
        self,
        grid_size: int = 15,
        num_resources: int = 8,
        num_obstacles: int = 15,
        resource_respawn_rate: float = 0.05,
        obstacle_movement: bool = False,
        energy_decay_rate: float = 1.0,
        resource_energy_value: int = 25,
        collision_penalty: float = -15.0,
        max_steps: int = 500,
        seed: Optional[int] = None
    ):
        # Input validation
        if grid_size < 3:
            raise ValueError(f"grid_size must be at least 3, got {grid_size}")
        if num_resources < 0:
            raise ValueError(f"num_resources must be non-negative, got {num_resources}")
        if num_obstacles < 0:
            raise ValueError(f"num_obstacles must be non-negative, got {num_obstacles}")
        if not 0 <= resource_respawn_rate <= 1:
            raise ValueError(f"resource_respawn_rate must be in [0,1], got {resource_respawn_rate}")
        if energy_decay_rate < 0:
            raise ValueError(f"energy_decay_rate must be non-negative, got {energy_decay_rate}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        
        # Check grid capacity
        max_cells = grid_size ** 3
        if num_resources + num_obstacles >= max_cells:
            raise ValueError(f"Too many objects for grid. Maximum {max_cells - 1} allowed.")
        
        # Environment parameters
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.num_obstacles = num_obstacles
        self.resource_respawn_rate = resource_respawn_rate
        self.obstacle_movement = obstacle_movement
        self.energy_decay_rate = energy_decay_rate
        self.resource_energy_value = resource_energy_value
        self.collision_penalty = collision_penalty
        self.max_steps = max_steps
        self.seed = seed
        
        # Random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Grid: 0=empty, 1=resource, -1=obstacle
        self.grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        
        # Dynamic objects
        self.resource_positions: List[Tuple[int, int, int]] = []
        self.obstacle_positions: List[Tuple[int, int, int]] = []
        self.obstacle_velocities: List[np.ndarray] = []
        
        # Agent state
        self.agent_pos: Optional[np.ndarray] = None
        self.agent_energy: float = 100.0
        self.step_count: int = 0
        
        # Statistics
        self.stats = {
            'resources_collected': 0,
            'obstacles_hit': 0,
            'total_distance_traveled': 0.0,
            'last_position': None
        }
        
        # Action space: continuous 3D movement vector
        self.action_space = ActionSpace(low=-1.0, high=1.0, shape=(3,))
        
        # Observation space: position + local_view + energy + min_distance
        obs_dim = 3 + 27 + 1 + 1
        self.observation_space = ObservationSpace(
            low=-1.0,
            high=float(grid_size),
            shape=(obs_dim,)
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        # Clear grid
        self.grid.fill(0)
        
        # Place resources
        self.resource_positions = []
        attempts = 0
        while len(self.resource_positions) < self.num_resources and attempts < 1000:
            pos = tuple(np.random.randint(0, self.grid_size, size=3))
            if pos not in self.resource_positions and self.grid[pos] == 0:
                self.resource_positions.append(pos)
                self.grid[pos] = 1.0
            attempts += 1
        
        if len(self.resource_positions) < self.num_resources:
            raise RuntimeError("Failed to place all resources")
        
        # Place obstacles
        self.obstacle_positions = []
        self.obstacle_velocities = []
        attempts = 0
        while len(self.obstacle_positions) < self.num_obstacles and attempts < 1000:
            pos = tuple(np.random.randint(0, self.grid_size, size=3))
            if pos not in self.resource_positions and pos not in self.obstacle_positions:
                self.obstacle_positions.append(pos)
                self.grid[pos] = -1.0
                if self.obstacle_movement:
                    vel = np.random.uniform(-0.3, 0.3, size=3)
                    self.obstacle_velocities.append(vel)
            attempts += 1
        
        if len(self.obstacle_positions) < self.num_obstacles:
            raise RuntimeError("Failed to place all obstacles")
        
        # Place agent at random valid position
        attempts = 0
        while attempts < 1000:
            pos = tuple(np.random.randint(0, self.grid_size, size=3))
            if self.grid[pos] == 0:
                self.agent_pos = np.array(pos, dtype=np.float32)
                break
            attempts += 1
        
        if self.agent_pos is None:
            raise RuntimeError("Failed to find valid starting position")
        
        # Reset state variables
        self.agent_energy = 100.0
        self.step_count = 0
        
        # Reset statistics
        self.stats = {
            'resources_collected': 0,
            'obstacles_hit': 0,
            'total_distance_traveled': 0.0,
            'last_position': tuple(self.agent_pos.astype(int))
        }
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if self.agent_pos is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")
        
        # Validate and normalize action
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        if action.shape != (3,):
            raise ValueError(f"Action must have shape (3,), got {action.shape}")
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        self.step_count += 1
        
        # Compute new position
        movement_scale = 1.5
        movement = action * movement_scale
        new_pos = self.agent_pos + movement
        
        # Apply boundary constraints
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        new_pos_int = new_pos.astype(int)
        
        # Compute Euclidean distance traveled
        distance = np.linalg.norm(new_pos - self.agent_pos)
        self.stats['total_distance_traveled'] += distance
        
        # Initialize reward and state
        reward = -0.1
        done = False
        truncated = False
        info = {}
        
        # Check grid cell value at new position
        grid_value = self.grid[tuple(new_pos_int)]
        
        # Resource collection
        if grid_value == 1.0:
            reward = self.resource_energy_value
            self.agent_energy += self.resource_energy_value
            self.grid[tuple(new_pos_int)] = 0.0
            resource_tuple = tuple(new_pos_int)
            if resource_tuple in self.resource_positions:
                self.resource_positions.remove(resource_tuple)
            self.stats['resources_collected'] += 1
            info['resource_collected'] = True
            self.agent_pos = new_pos.copy()
        
        # Obstacle collision
        elif grid_value == -1.0:
            reward = self.collision_penalty
            self.agent_energy += self.collision_penalty
            self.stats['obstacles_hit'] += 1
            info['obstacle_hit'] = True
            # Agent cannot move into obstacle
            new_pos = self.agent_pos.copy()
            new_pos_int = self.agent_pos.astype(int)
        
        else:  # Empty space
            self.agent_pos = new_pos.copy()
        
        # Energy dynamics
        self.agent_energy -= self.energy_decay_rate
        
        # Check terminal condition
        if self.agent_energy <= 0:
            done = True
            reward -= 10.0  # Additional penalty for energy depletion
        
        # Check truncation
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Reward shaping: proximity reward
        if len(self.resource_positions) > 0:
            distances_to_resources = [
                np.linalg.norm(self.agent_pos - np.array(rp))
                for rp in self.resource_positions
            ]
            min_distance = min(distances_to_resources)
            proximity_reward = 1.0 / (min_distance + 1.0)
            reward += proximity_reward * 0.1
        
        # Update moving obstacles
        if self.obstacle_movement:
            self._update_moving_obstacles()
        
        # Resource respawn
        if np.random.random() < self.resource_respawn_rate and len(self.resource_positions) < self.num_resources:
            self._respawn_resource()
        
        # Update statistics
        self.stats['last_position'] = tuple(self.agent_pos.astype(int))
        
        info.update({
            'energy': float(self.agent_energy),
            'step': self.step_count,
            'resources_remaining': len(self.resource_positions)
        })
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        pos = self.agent_pos
        
        # Normalized position
        pos_normalized = pos / self.grid_size
        
        # Local view: 3x3x3 cube around agent
        local_view = np.zeros(27, dtype=np.float32)
        idx = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    check_pos = (pos.astype(int) + np.array([dx, dy, dz]))
                    if (0 <= check_pos[0] < self.grid_size and
                        0 <= check_pos[1] < self.grid_size and
                        0 <= check_pos[2] < self.grid_size):
                        local_view[idx] = self.grid[tuple(check_pos)]
                    else:
                        local_view[idx] = -1.0
                    idx += 1
        
        # Normalized energy
        energy_normalized = self.agent_energy / 100.0
        
        # Normalized minimum distance to resource
        if len(self.resource_positions) > 0:
            distances = [
                np.linalg.norm(pos - np.array(rp))
                for rp in self.resource_positions
            ]
            min_dist = min(distances) / (self.grid_size * np.sqrt(3))
        else:
            min_dist = 1.0
        
        # Concatenate all components
        obs = np.concatenate([
            pos_normalized,
            local_view,
            [energy_normalized, min_dist]
        ])
        
        return obs.astype(np.float32)
    
    def _update_moving_obstacles(self):
        """Update positions of moving obstacles"""
        for i, (pos, vel) in enumerate(zip(self.obstacle_positions, self.obstacle_velocities)):
            old_pos = pos
            new_pos = np.array(pos) + vel
            
            # Boundary constraints
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            new_pos_int = tuple(new_pos.astype(int))
            
            # Check collision with other obstacles
            if new_pos_int not in self.obstacle_positions:
                self.grid[old_pos] = 0.0
                self.grid[new_pos_int] = -1.0
                self.obstacle_positions[i] = new_pos_int
                
                # Elastic collision with boundaries
                if new_pos_int[0] == 0 or new_pos_int[0] == self.grid_size - 1:
                    self.obstacle_velocities[i][0] *= -1
                if new_pos_int[1] == 0 or new_pos_int[1] == self.grid_size - 1:
                    self.obstacle_velocities[i][1] *= -1
                if new_pos_int[2] == 0 or new_pos_int[2] == self.grid_size - 1:
                    self.obstacle_velocities[i][2] *= -1
    
    def _respawn_resource(self):
        """Respawn a resource at random empty location"""
        attempts = 0
        while attempts < 100:
            pos = tuple(np.random.randint(0, self.grid_size, size=3))
            if self.grid[pos] == 0 and pos != tuple(self.agent_pos.astype(int)):
                self.resource_positions.append(pos)
                self.grid[pos] = 1.0
                break
            attempts += 1
    
    def get_stats(self) -> Dict:
        """Return environment statistics"""
        return self.stats.copy()
    
    def render(self, mode: str = 'human') -> Optional[Dict]:
        """Render environment state"""
        if mode == 'human' and self.agent_pos is not None:
            return {
                'agent_pos': tuple(self.agent_pos.astype(int)),
                'agent_energy': self.agent_energy,
                'resource_positions': self.resource_positions.copy(),
                'obstacle_positions': self.obstacle_positions.copy(),
                'grid_size': self.grid_size
            }
        return None
