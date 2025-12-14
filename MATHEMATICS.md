# Mathematical Foundations

This document covers the math behind everything in this project. All algorithms are derived from first principles.

## Neural Networks

### Forward Pass

For layer l:
```
z^(l) = W^(l) · a^(l-1) + b^(l)
a^(l) = σ(z^(l))
```

Where W is the weight matrix, b is bias, σ is activation, and a^(0) is the input.

### Activation Functions

**ReLU:**
```
σ(x) = max(0, x)
σ'(x) = 1 if x > 0, else 0
```

**Tanh:**
```
σ(x) = tanh(x)
σ'(x) = 1 - tanh²(x)
```

**Sigmoid:**
```
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x)(1 - σ(x))
```

### Backpropagation

Output layer:
```
δ^(L) = ∇_a C ⊙ σ'(z^(L))
```

Hidden layers:
```
δ^(l) = ((W^(l+1))^T · δ^(l+1)) ⊙ σ'(z^(l))
```

Gradients:
```
∇_W^(l) C = a^(l-1) · (δ^(l))^T
∇_b^(l) C = δ^(l)
```

### Adam Optimizer

Moment estimates:
```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
```

Bias correction:
```
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
```

Update:
```
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

With β₁=0.9, β₂=0.999, ε=10⁻⁸.

## PPO Algorithm

### Policy Gradient

Objective:
```
J(θ) = E_{τ~π_θ} [R(τ)]
```

Gradient:
```
∇_θ J(θ) = E_{τ~π_θ} [∇_θ log π_θ(a|s) · R(τ)]
```

### Clipped Objective

PPO uses:
```
L^CLIP(θ) = E_t [min(r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t)]
```

Where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the importance ratio, and ε=0.2.

### GAE (Generalized Advantage Estimation)

TD error:
```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

GAE:
```
Â_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

With γ=0.99 (discount) and λ=0.95 (GAE parameter).

### Value Function Loss

```
L^VF(φ) = E_t [(V_φ(s_t) - R_t)²]
```

Where R_t is the discounted return.

### Entropy Bonus

For Gaussian policy:
```
H(π) = 0.5 · Σ_i log(2πe · σ_i²)
```

Encourages exploration by maximizing entropy.

## Environment

### State Space

State vector:
```
s_t = [p_t, L_t, E_t, d_min_t]
```

- p_t: normalized position [0,1]³
- L_t: local 3×3×3 grid view (27 values)
- E_t: normalized energy [0,1]
- d_min_t: normalized min distance to resource [0,1]

### Action Space

Continuous 3D movement:
```
a_t ∈ [-1, 1]³
```

### Transition

Position update:
```
p_{t+1} = clip(p_t + α · a_t, [0, grid_size-1]³)
```

With movement scale α=1.5.

### Reward Function

```
R = R_resource · I(resource) + R_obstacle · I(obstacle) 
  + R_proximity · (1/(d_min + 1)) - R_move - R_energy
```

Where:
- R_resource = 25
- R_obstacle = -15
- R_proximity = 0.1
- R_move = 0.1
- R_energy = energy_decay_rate

### Energy Dynamics

```
E_{t+1} = E_t - β + R_resource · I(resource_collected)
```

With decay rate β=1.0.

## Gaussian Policy

### Policy Distribution

```
π_θ(a|s) = N(μ_θ(s), Σ_θ(s))
```

Where μ is the mean (network output) and Σ is diagonal covariance (learnable log_std).

### Log Probability

```
log π_θ(a|s) = -0.5 · Σ_i [((a_i - μ_i)² / σ_i²) + log(2πσ_i²)]
```

### Gradients

Gradient w.r.t. mean:
```
∇_μ log π_θ(a|s) = (a - μ) / σ²
```

Gradient w.r.t. log_std:
```
∇_{log_σ} log π_θ(a|s) = ((a - μ)² / σ²) - 1
```

## Communication

### Message Range

Message received if:
```
d(p_sender, p_receiver) ≤ range_limit AND (t_current - t_sent) ≤ TTL
```

Where d is Euclidean distance.

### Information Sharing

Resource info: position, energy_value, confidence, timestamp
Obstacle info: position, severity, timestamp

## Optimization Details

### Gradient Clipping

```
g_clipped = g · min(1, max_norm / ||g||₂)
```

Prevents exploding gradients.

### Batch Normalization

```
x̂ = (x - μ) / √(σ² + ε)
y = γ · x̂ + β
```

With learnable γ and β.

### Dropout

During training:
```
y = x · m / (1 - p)
```

Where m ~ Bernoulli(1-p).

## Numerical Stability

- Clip log probabilities: [-10, 10]
- Add epsilon: σ² + 10⁻⁸
- Normalize advantages: (Â - μ(Â)) / (σ(Â) + ε)

### Initialization

He initialization:
```
W ~ N(0, √(2 / n_in))
```

Good for ReLU activations.

---

All implementations follow these formulations exactly.
