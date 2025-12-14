# Mathematical Foundations

Complete mathematical derivations for all algorithms in this implementation.

## Neural Networks

### Forward Propagation

For layer $l$ with input $a^{(l-1)}$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \sigma(z^{(l)})
$$

where $W^{(l)} \in \mathbb{R}^{n_{\text{in}} \times n_{\text{out}}}$ is the weight matrix, $b^{(l)} \in \mathbb{R}^{n_{\text{out}}}$ is the bias vector, and $\sigma$ is the activation function.

### Activation Functions

**ReLU:**

$$
\sigma(x) = \max(0, x)
$$

$$
\sigma'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}
$$

**Tanh:**

$$
\sigma(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

$$
\sigma'(x) = 1 - \tanh^2(x)
$$

**Sigmoid:**

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**Linear:**

$$
\sigma(x) = x
$$

$$
\sigma'(x) = 1
$$

### Backpropagation

Error signal for output layer $L$:

$$
\delta^{(L)} = \nabla_a C \odot \sigma'(z^{(L)})
$$

Error signal for hidden layer $l$:

$$
\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})
$$

Gradients with respect to parameters:

$$
\nabla_{W^{(l)}} C = a^{(l-1)} (\delta^{(l)})^T
$$

$$
\nabla_{b^{(l)}} C = \delta^{(l)}
$$

### Batch Normalization

Forward pass during training:

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

where $m$ is batch size, $\gamma$ and $\beta$ are learnable parameters, and $\epsilon = 10^{-8}$.

Running statistics (inference):

$$
\mu_{\text{running}} = \text{momentum} \cdot \mu_{\text{running}} + (1 - \text{momentum}) \cdot \mu_B
$$

$$
\sigma_{\text{running}}^2 = \text{momentum} \cdot \sigma_{\text{running}}^2 + (1 - \text{momentum}) \cdot \sigma_B^2
$$

Backward pass gradients:

$$
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \hat{x}_i
$$

$$
\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}
$$

$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \gamma
$$

$$
\frac{\partial L}{\partial \sigma_B^2} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} (x_i - \mu_B) \cdot \left(-\frac{1}{2}\right) (\sigma_B^2 + \epsilon)^{-3/2}
$$

$$
\frac{\partial L}{\partial \mu_B} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{-2}{m} \sum_{i=1}^{m} (x_i - \mu_B)
$$

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}
$$

### Dropout

During training:

$$
y_i = \begin{cases} \frac{x_i}{1-p} & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}
$$

where $p$ is the dropout rate. The scaling factor $1/(1-p)$ maintains expected activation during training.

During inference:

$$
y_i = x_i
$$

### Adam Optimizer

First moment estimate:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

Second moment estimate:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Bias correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Parameter update:

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and $\alpha$ is the learning rate.

### Weight Initialization

He initialization for ReLU (uniform distribution):

$$
W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)
$$

where $n_{\text{in}}$ is the number of input units.

## PPO Algorithm

### Policy Gradient Objective

The policy gradient objective:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory and $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ is the discounted return.

Policy gradient theorem:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)\right]
$$

### Clipped Surrogate Objective

PPO uses a clipped objective to prevent large policy updates:

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the importance sampling ratio, $\hat{A}_t$ is the advantage estimate, and $\epsilon = 0.2$ is the clipping parameter.

The gradient uses the term that minimizes the objective:
$$
\nabla_\theta L^{CLIP}(\theta) = \mathbb{E}_t\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \begin{cases} \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t & \text{if } \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t < r_t(\theta) \hat{A}_t \\ r_t(\theta) \hat{A}_t & \text{otherwise} \end{cases}\right]
$$

### Generalized Advantage Estimation (GAE)

Temporal difference error:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE advantage:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\gamma = 0.99$ is the discount factor and $\lambda = 0.95$ is the GAE parameter.

Returns:

$$
\hat{R}_t = \hat{A}_t + V(s_t)
$$

Advantage normalization:

$$
\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + \epsilon}
$$

where $\mu(\hat{A})$ and $\sigma(\hat{A})$ are the mean and standard deviation of advantages in the batch.

### Value Function Loss

$$
L^{VF}(\phi) = c_v \mathbb{E}_t\left[(V_\phi(s_t) - \hat{R}_t)^2\right]
$$

where $c_v = 0.5$ is the value coefficient and $\phi$ are the value network parameters.

### Entropy Bonus

For a Gaussian policy with diagonal covariance:

$$
H(\pi_\theta) = \frac{1}{2} \sum_{i=1}^{d} \log(2\pi e \sigma_i^2)
$$

where $d$ is the action dimension and $\sigma_i$ is the standard deviation for dimension $i$.

The entropy bonus is added to the policy loss:

$$
L(\theta) = L^{CLIP}(\theta) - c_e H(\pi_\theta)
$$

where $c_e = 0.01$ is the entropy coefficient.

### Gaussian Policy

Policy distribution:

$$
\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))
$$

where $\mu_\theta(s)$ is the mean (network output) and $\Sigma_\theta(s) = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$ is the diagonal covariance matrix.

Log probability:

$$
\log \pi_\theta(a|s) = -\frac{1}{2}\sum_{i=1}^{d}\left[\frac{(a_i - \mu_i)^2}{\sigma_i^2} + \log(2\pi \sigma_i^2)\right]
$$

Gradient with respect to mean:

$$
\nabla_\mu \log \pi_\theta(a|s) = \frac{a - \mu}{\sigma^2}
$$

Gradient with respect to log standard deviation:

$$
\nabla_{\log \sigma} \log \pi_\theta(a|s) = \frac{(a - \mu)^2}{\sigma^2} - 1
$$

### Gradient Clipping

Gradients are clipped to prevent exploding gradients:

$$
g_{\mathrm{clipped}} = g \cdot \min\left(1, \frac{\mathrm{max\_norm}}{\left\|g\right\|_2}\right)
$$

where $\mathrm{max\_norm} = 0.5$ is the maximum gradient norm.

## Environment

### State Space

The observation vector:

$$
s_t = [p_t, L_t, E_t, d_{\text{min},t}]
$$

where:
- $p_t \in [0,1]^3$: normalized position $(x/\mathrm{grid\_size}, y/\mathrm{grid\_size}, z/\mathrm{grid\_size})$ where $\mathrm{grid\_size}$ is the environment grid size
- $L_t \in \mathbb{R}^{27}$: local 3×3×3 grid view around agent position
- $E_t \in [0,1]$: normalized energy level $E_t / 100$
- $d_{\text{min},t} \in [0,1]$: normalized minimum distance to nearest resource

### Action Space

Continuous 3D movement:

$$
a_t \in [-1, 1]^3
$$

### State Transition

Position update:

$$
p_{t+1} = \text{clip}(p_t + \alpha \cdot a_t, [0, \mathrm{grid\_size} - 1]^3)
$$

where $\alpha = 1.5$ is the movement scale.

### Reward Function

$$
R_t = R_{\text{resource}} \cdot \mathbb{I}(\text{resource}) + R_{\text{obstacle}} \cdot \mathbb{I}(\text{obstacle}) + R_{\text{proximity}} \cdot \frac{1}{d_{\text{min}} + 1} - R_{\text{move}} - R_{\text{energy}}
$$

where:
- $R_{\text{resource}} = 25$: reward for collecting resource
- $R_{\text{obstacle}} = -15$: penalty for hitting obstacle
- $R_{\text{proximity}} = 0.1$: proximity reward coefficient
- $R_{\text{move}} = 0.1$: movement penalty per step
- $R_{\text{energy}} = 0$: energy decay is handled separately

Terminal penalty:

$$
R_{\text{terminal}} = -10 \quad \text{if } E_t \leq 0
$$

### Energy Dynamics

$$
E_{t+1} = E_t - \beta + R_{\text{resource}} \cdot \mathbb{I}(\text{resource collected}) + R_{\text{obstacle}} \cdot \mathbb{I}(\text{obstacle hit})
$$

where $\beta = 1.0$ is the energy decay rate per step.

Terminal condition:

$$
\text{done} = \begin{cases} \text{True} & \text{if } E_{t+1} \leq 0 \\ \text{False} & \text{otherwise} \end{cases}
$$

### Resource Respawn

Resources respawn probabilistically:

$$
\text{respawn} \sim \text{Bernoulli}(p_{\text{respawn}})
$$

where $p_{\text{respawn}} = 0.05$ is the respawn rate per step, and respawn occurs only if the number of resources is below the initial count.

## Communication

### Message Range

A message is received if:

$$
\left\|p_{\text{sender}} - p_{\text{receiver}}\right\|_2 \leq \mathrm{range\_limit} \quad \text{AND} \quad (t_{\text{current}} - t_{\text{sent}}) \leq \text{TTL}
$$

where $\left\|\cdot\right\|_2$ is the Euclidean distance, $\mathrm{range\_limit}$ depends on message type, and $\mathrm{TTL} = 50$ is the time-to-live.

### Cooperation History

For mixed cooperation mode, cooperation score:

$$
\mathrm{score}_j = \frac{\mathrm{helpful\_actions}_j \cdot \mathrm{decay}^{t - t_{\mathrm{last}}}}{\mathrm{total\_interactions}_j \cdot \mathrm{decay}^{t - t_{\mathrm{last}}}}
$$

where $\text{decay} = 0.95$ is the decay factor and $j$ is the other agent ID.

Cooperation decision:

$$
\text{cooperate} = \begin{cases} \text{True} & \text{if } \text{score}_j \geq 0.5 \\ \text{False} & \text{otherwise} \end{cases}
$$

## Numerical Stability

### Log Probability Clipping

Log probability ratios are clipped before exponentiation:

$$
\log r_t = \text{clip}(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{\text{old}}}(a_t|s_t), -10, 10)
$$

### Variance Regularization

Variance terms include epsilon for numerical stability:

$$
\sigma^2 \leftarrow \sigma^2 + 10^{-8}
$$

### Advantage Normalization

Advantages are normalized to zero mean and unit variance:

$$
\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + 10^{-8}}
$$

## Implementation Details

### PPO Update Procedure

1. Collect trajectories: store $(s_t, a_t, r_t, V(s_t), \log \pi(a_t|s_t), \text{done}_t)$ in buffer
2. Compute advantages: use GAE to compute $\hat{A}_t$ and $\hat{R}_t$ for all timesteps
3. Normalize advantages: subtract mean and divide by standard deviation
4. Multiple epochs: for $K = 10$ epochs:
   - Sample random batch from buffer
   - Compute policy loss $L^{CLIP}(\theta)$
   - Compute value loss $L^{VF}(\phi)$
   - Compute entropy $H(\pi_\theta)$
   - Backpropagate gradients through policy and value networks
   - Update parameters using Adam optimizer
5. Clear buffer: reset for next collection phase

### Network Architecture

Policy network (actor):
- Input: observation $s_t$ (dimension varies with communication features)
- Hidden layers: $[128, 128, 64]$ with tanh activation
- Output: mean $\mu(s_t)$ (action dimension)
- Separate learnable $\log \sigma$ parameter (not network output)

Value network (critic):
- Input: observation $s_t$
- Hidden layers: $[64, 64]$ with tanh activation
- Output: value estimate $V(s_t)$ (scalar)

### Hyperparameters

PPO:
- Discount factor: $\gamma = 0.99$
- GAE parameter: $\lambda = 0.95$
- Clipping parameter: $\epsilon = 0.2$
- Value coefficient: $c_v = 0.5$
- Entropy coefficient: $c_e = 0.01$
- Max gradient norm: $0.5$
- Update epochs: $K = 10$
- Batch size: $64$
- Buffer size: $2048$

Optimizer:
- Learning rate: $\alpha = 3 \times 10^{-4}$ for policy and value networks
- Adam $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

Environment:
- Grid size: $15 \times 15 \times 15$
- Movement scale: $\alpha = 1.5$
- Energy decay: $\beta = 1.0$ per step
- Resource value: $25$
- Collision penalty: $-15$
- Max steps per episode: $500$

---

All implementations follow these mathematical formulations exactly.
