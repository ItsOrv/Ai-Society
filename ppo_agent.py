"""
PPO (Proximal Policy Optimization) algorithm implementation
Includes Actor-Critic architecture
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from neural_network import NeuralNetwork, ActivationFunction


class PPOBuffer:
    """Buffer for storing experiences in PPO"""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
    
    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, 
              value: float, log_prob: float, done: bool):
        """Store an experience"""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_advantages_and_returns(
        self, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
        last_done: bool = False
    ):
        """Compute advantages and returns using GAE"""
        advantages = np.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        self.returns = advantages + self.values
        self.advantages = advantages
        
        if self.size > 0:
            self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-8)
    
    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Get a random batch from buffer"""
        if self.size == 0:
            raise ValueError("Buffer is empty, cannot get batch")
        actual_batch_size = min(batch_size, self.size)
        if actual_batch_size == 0:
            raise ValueError("Cannot create batch of size 0")
        indices = np.random.choice(self.size, size=actual_batch_size, replace=False)
        return {
            'obs': self.obs[indices],
            'actions': self.actions[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices],
            'old_log_probs': self.log_probs[indices]
        }
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0


class GaussianPolicy:
    """Gaussian policy for continuous action space"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = [64, 64]):
        layer_sizes = [obs_dim] + hidden_sizes + [action_dim]
        self.mean_network = NeuralNetwork(
            layer_sizes=layer_sizes,
            activations=['tanh'] * (len(hidden_sizes)) + ['linear'],
            learning_rate=3e-4
        )
        
        self.log_std = np.zeros(action_dim, dtype=np.float32) - 0.5
        self.log_std_grad = np.zeros(action_dim, dtype=np.float32)
    
    def forward(self, obs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and log_prob"""
        mean = self.mean_network.forward(obs, training=training)
        std_base = np.exp(self.log_std)
        if len(mean.shape) > 1:
            std = std_base.reshape(1, -1)
        else:
            std = std_base
        return mean, std
    
    def sample(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from Gaussian distribution"""
        mean, std = self.forward(obs, training=True)
        noise = np.random.normal(0, 1, size=mean.shape)
        action = mean + std * noise
        
        log_prob = self._gaussian_log_prob(action, mean, std)
        
        return action, log_prob
    
    def _gaussian_log_prob(self, action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Compute log probability for Gaussian distribution"""
        var = std ** 2
        log_prob = -0.5 * (((action - mean) ** 2) / (var + 1e-8) + np.log(2 * np.pi * var + 1e-8))
        return np.sum(log_prob, axis=-1)
    
    def log_prob(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute log probability for given actions"""
        mean, std = self.forward(obs, training=True)
        return self._gaussian_log_prob(actions, mean, std)
    
    def update(self):
        """Update network"""
        self.mean_network.update()


class ValueNetwork:
    """Critic network for value function estimation"""
    
    def __init__(self, obs_dim: int, hidden_sizes: List[int] = [64, 64]):
        layer_sizes = [obs_dim] + hidden_sizes + [1]
        self.network = NeuralNetwork(
            layer_sizes=layer_sizes,
            activations=['tanh'] * (len(hidden_sizes)) + ['linear'],
            learning_rate=3e-4
        )
    
    def forward(self, obs: np.ndarray, training: bool = True) -> np.ndarray:
        """Estimate value"""
        return self.network.forward(obs, training=training).squeeze(-1)
    
    def update(self):
        """Update network"""
        self.network.update()


class PPOAgent:
    """Agent with complete PPO algorithm"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [64, 64],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_sizes)
        self.value_net = ValueNetwork(obs_dim, hidden_sizes)
        
        self.buffer = PPOBuffer(buffer_size, obs_dim, action_dim)
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'explained_variance': []
        }
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action"""
        obs = obs.reshape(1, -1) if len(obs.shape) == 1 else obs
        
        if deterministic:
            mean, _ = self.policy.forward(obs, training=False)
            action = mean
            log_prob = self.policy.log_prob(obs, action)
        else:
            action, log_prob = self.policy.sample(obs)
        
        value = self.value_net.forward(obs, training=False)
        
        return action[0], log_prob[0], value[0]
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in buffer"""
        obs_flat = obs.flatten() if isinstance(obs, dict) else obs
        self.buffer.store(obs_flat, action, reward, value, log_prob, done)
    
    def update(self, last_obs: np.ndarray = None, last_done: bool = False):
        """Update policy and value network"""
        if self.buffer.size == 0:
            return
        
        last_value = 0.0
        if last_obs is not None:
            last_obs_flat = last_obs.flatten() if isinstance(last_obs, dict) else last_obs
            last_value = self.value_net.forward(last_obs_flat.reshape(1, -1), training=False)[0]
        
        self.buffer.compute_advantages_and_returns(
            self.gamma, self.gae_lambda, last_value, last_done
        )
        
        for epoch in range(self.update_epochs):
            batch = self.buffer.get_batch(self.batch_size)
            self._update_step(batch)
        
        self.buffer.clear()
    
    def _update_step(self, batch: Dict[str, np.ndarray]):
        """One update step"""
        obs = batch['obs']
        actions = batch['actions']
        advantages = batch['advantages']
        returns = batch['returns']
        old_log_probs = batch['old_log_probs']
        
        mean, std = self.policy.forward(obs, training=True)
        new_log_probs = self.policy.log_prob(obs, actions)
        log_ratio = new_log_probs - old_log_probs
        ratio = np.exp(np.clip(log_ratio, -10, 10))
        
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        clipped_policy_loss = np.mean(np.maximum(policy_loss_1, policy_loss_2))
        
        if len(std.shape) > 1:
            entropy_per_sample = 0.5 * np.sum(np.log(2 * np.pi * np.e * std ** 2 + 1e-8), axis=-1)
        else:
            entropy_per_sample = 0.5 * np.sum(np.log(2 * np.pi * np.e * std ** 2 + 1e-8))
        entropy = np.mean(entropy_per_sample)
        
        policy_loss = clipped_policy_loss - self.entropy_coef * entropy
        
        values = self.value_net.forward(obs, training=True)
        value_loss = self.value_coef * np.mean((values - returns) ** 2)
        
        unclipped_term = ratio * advantages
        clipped_term = clipped_ratio * advantages
        use_clipped = clipped_term < unclipped_term
        
        policy_grad_log_prob = np.where(
            use_clipped,
            -advantages * clipped_ratio,
            -advantages * ratio
        )
        
        var = std ** 2 + 1e-8
        if len(actions.shape) > 1 and len(mean.shape) > 1:
            grad_mean = policy_grad_log_prob.reshape(-1, 1) * (actions - mean) / var
        else:
            if len(actions.shape) == 1:
                grad_mean = policy_grad_log_prob * (actions - mean) / var
            else:
                grad_mean = policy_grad_log_prob.reshape(-1, 1) * (actions - mean) / var
        
        if len(grad_mean.shape) == 1:
            grad_mean = grad_mean.reshape(1, -1)
        
        grad_norm = np.linalg.norm(grad_mean)
        if grad_norm > self.max_grad_norm:
            grad_mean = grad_mean * (self.max_grad_norm / (grad_norm + 1e-8))
        
        self.policy.mean_network.backward(grad_mean)
        
        value_error = values - returns
        batch_size = len(values) if len(values) > 0 else 1
        value_grad = self.value_coef * 2.0 * value_error / batch_size
        if len(value_grad.shape) == 1:
            value_grad = value_grad.reshape(-1, 1)
        self.value_net.network.backward(value_grad)
        
        self.policy.mean_network.update()
        self.value_net.network.update()
        
        var = std ** 2 + 1e-8
        if len(actions.shape) > 1 and len(mean.shape) > 1:
            log_std_grad_policy = (
                policy_grad_log_prob.reshape(-1, 1) * 
                (((actions - mean) ** 2) / var - 1.0)
            ).mean(axis=0)
            entropy_grad = self.entropy_coef * (1.0 - std ** 2).mean(axis=0)
        else:
            if len(actions.shape) == 1:
                log_std_grad_policy = policy_grad_log_prob * (
                    ((actions - mean) ** 2) / var - 1.0
                )
                entropy_grad = self.entropy_coef * (1.0 - std ** 2)
            else:
                log_std_grad_policy = (
                    policy_grad_log_prob.reshape(-1, 1) * 
                    (((actions - mean) ** 2) / var - 1.0)
                )
                if len(log_std_grad_policy.shape) > 1:
                    log_std_grad_policy = log_std_grad_policy.squeeze(axis=0) if log_std_grad_policy.shape[0] == 1 else log_std_grad_policy.mean(axis=0)
                entropy_grad = self.entropy_coef * (1.0 - std ** 2)
                if len(entropy_grad.shape) > 1:
                    entropy_grad = entropy_grad.squeeze(axis=0) if entropy_grad.shape[0] == 1 else entropy_grad.mean(axis=0)
        
        log_std_grad = log_std_grad_policy + entropy_grad
        if len(log_std_grad.shape) == 0:
            log_std_grad = np.array([log_std_grad] * self.action_dim)
        elif len(log_std_grad.shape) > 0 and log_std_grad.shape[0] == self.policy.log_std.shape[0]:
            pass
        else:
            if log_std_grad.shape[0] > self.action_dim:
                log_std_grad = log_std_grad[:self.action_dim]
            else:
                log_std_grad = np.pad(log_std_grad, (0, self.action_dim - log_std_grad.shape[0]), 'constant')
        
        self.policy.log_std += 0.001 * log_std_grad
        self.policy.log_std = np.clip(self.policy.log_std, -2.0, 1.0)
        
        explained_var = 1 - np.var(returns - values) / (np.var(returns) + 1e-8)
        self.training_stats['policy_loss'].append(float(policy_loss))
        self.training_stats['value_loss'].append(float(value_loss))
        self.training_stats['entropy'].append(float(entropy))
        self.training_stats['explained_variance'].append(float(explained_var))
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        if not self.training_stats['policy_loss']:
            return {}
        
        return {
            'policy_loss': np.mean(self.training_stats['policy_loss'][-100:]),
            'value_loss': np.mean(self.training_stats['value_loss'][-100:]),
            'entropy': np.mean(self.training_stats['entropy'][-100:]),
            'explained_variance': np.mean(self.training_stats['explained_variance'][-100:])
        }
    
    def save(self, filepath: str):
        """Save model"""
        pass
    
    def load(self, filepath: str):
        """Load model"""
        pass
