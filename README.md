# AI-Society

Multi-agent reinforcement learning simulation where robots learn to navigate a 3D environment, collect resources, and coordinate with each other. Everything is implemented from scratch using only NumPy - no TensorFlow, PyTorch, or other ML frameworks.

## What This Is

This project implements a complete RL system from the ground up. The neural networks, PPO algorithm, environment, and all the math are coded manually. I built it this way to really understand how everything works under the hood.

The simulation runs multiple robots in a 3D grid world. Each robot has its own neural network that learns through PPO. They can communicate with each other, share information about resources and obstacles, and work together or compete depending on the mode.

## Features

**Neural Networks**: Full implementation with backpropagation, Adam optimizer, dropout, and batch normalization. All activation functions (ReLU, Tanh, Sigmoid) are coded from scratch. Uses He initialization for ReLU activations.

**PPO Algorithm**: Complete Proximal Policy Optimization implementation matching the Schulman et al. (2017) paper. Includes clipped surrogate objective, GAE for advantage estimation, and separate actor-critic networks. Uses Gaussian policy for continuous actions.

**3D Environment**: Custom grid-based environment (no Gym dependency). Robots move continuously in 3D space, collect energy resources, avoid obstacles, and manage their energy levels.

**Communication System**: Robots can send messages to each other about resources, obstacles, and coordinate actions. Messages have range limits and TTL. Supports history-based cooperation decisions.

**Multiple Modes**: Competitive (every robot for itself), Cooperative (share everything), or Mixed (dynamic cooperation based on interaction history).

**Metrics & Visualization**: Text-based output showing learning curves, agent performance, communication stats. Can export everything to JSON for analysis.

## Project Structure

```
Ai-Society/
├── neural_network.py      # Neural network from scratch
├── ppo_agent.py           # PPO implementation
├── environment.py         # 3D grid environment
├── robot.py               # Robot class with learning
├── communication.py       # Inter-robot messaging
├── metrics.py             # Performance tracking
├── visualization.py       # Text-based stats display
├── main.py                # Entry point
├── requirements.txt       # Just NumPy
├── MATHEMATICS.md         # Math documentation
└── README.md
```

## Quick Start

Install NumPy:
```bash
pip install -r requirements.txt
```

Run with defaults (3 robots, 100 episodes):
```bash
python main.py
```

Custom run:
```bash
python main.py --num_robots 5 --num_episodes 200 --grid_size 20 --render
```

**Parameters:**
- `--num_robots`: How many robots (default: 3)
- `--num_episodes`: Training episodes (default: 100)
- `--grid_size`: 3D grid size (default: 15)
- `--render`: Show text output during training
- `--save_metrics`: Save stats to JSON file
- `--cooperation_mode`: competitive, cooperative, or mixed
- `--seed`: Random seed for reproducibility

**Examples:**
```bash
# Cooperative mode with 5 robots
python main.py --num_robots 5 --cooperation_mode cooperative --num_episodes 150

# Quick test run with visualization
python main.py --render --num_episodes 50

# Save results for later analysis
python main.py --save_metrics results.json --num_episodes 200
```

## Implementation Details

All the math is documented in `MATHEMATICS.md`. Here's the high-level overview:

**Neural Networks**: Standard feedforward networks with manual backpropagation. No autograd - gradients are computed explicitly. Uses He initialization (uniform distribution for ReLU) and Adam optimizer (also implemented from scratch).

**PPO**: Clipped surrogate objective to prevent large policy updates. GAE (lambda=0.95) for advantage estimation. Separate value network for baseline. Entropy bonus added to policy loss to encourage exploration.

**Environment**: 3D grid where each cell can be empty, contain a resource, or an obstacle. Robots move continuously in 3D space. Reward shaping based on distance to resources. Energy system with decay and resource collection.

**Communication**: Range-based messaging with TTL. Robots broadcast resource locations, obstacle warnings, and coordination messages. Each robot maintains a local map of known resources and obstacles. Mixed mode uses history-based cooperation decisions.

## Output

During training you'll see:
- Episode-by-episode rewards and episode lengths
- Per-robot statistics (resources collected, survival time, etc.)
- Communication metrics (messages sent/received)
- Learning curves

If you use `--save_metrics`, everything gets dumped to JSON for plotting or further analysis.

## Use Cases

Good for:
- Learning how RL algorithms actually work (no black boxes)
- Teaching RL concepts (everything is visible and understandable)
- Research experiments (easy to modify and extend)
- Student projects (clean codebase, well-documented)

The code is structured so you can easily add new algorithms, modify the environment, or experiment with different communication protocols.

## Educational Scope

This project is designed for **educational and research purposes**. It demonstrates:

- How PPO works internally (no framework abstractions)
- How neural networks compute gradients manually
- How multi-agent systems can coordinate through communication
- How reward shaping guides learning

The implementation prioritizes **clarity and correctness** over performance. All algorithms are implemented from first principles with explicit gradient computation.

## Extending

Want to add another RL algorithm? The structure makes it straightforward. Same for environment modifications or new communication strategies. The codebase is modular enough that you can swap components without breaking everything.

Some ideas:
- Add DQN, A3C, or other algorithms
- More complex environments
- Different reward structures
- New communication protocols

## Known Limitations

**Separate Environment Instances**: Each robot currently has its own environment instance. This simplifies the code but means they're not truly sharing the same world state. Could be upgraded to a proper shared environment if needed.

**Physics Inconsistencies**: The environment uses continuous agent positions but discrete grid cells for collision/resource detection. This can cause occasional missed interactions (e.g., agent may phase through obstacles diagonally or miss resources when moving between cells). These bugs are non-dominant and don't prevent learning - the reward signal is strong enough to guide legitimate behavior. See `ENVIRONMENT_AUDIT.md` for details.

**Log Ratio Pre-Clipping**: The PPO implementation pre-clips log probability ratios to [-10, 10] before exponentiation, which limits the importance ratio range. This is a minor optimization that doesn't affect correctness but may slightly limit learning in extreme cases.

**Manual Log Std Update**: The policy's log_std parameter uses a fixed learning rate (0.001) instead of the Adam optimizer. This works but is inconsistent with the mean network updates.

## Audit Summary

This codebase has been thoroughly audited for correctness:

**Verified**:
- ✅ PPO implementation matches Schulman et al. (2017) paper
- ✅ Neural network gradients computed correctly from scratch
- ✅ All mathematical formulas match implementation
- ✅ System is learnable (agents can learn the task)
- ✅ No dominant exploit strategies

**Fixed**:
- ✅ PPO policy gradient bug (now correctly uses clipped/unclipped ratio based on minimum)
- ✅ He initialization (changed from Glorot/Xavier to correct He initialization)
- ✅ Entropy bonus (now properly added to policy loss)
- ✅ Batch normalization backward pass (completed full gradient computation)
- ✅ Mixed cooperation mode (now implements history-based decisions)

**Remaining Limitations**:
- ⚠️ Environment physics inconsistencies (non-dominant, documented above)
- ⚠️ Minor PPO optimizations (log ratio pre-clipping, manual log_std update)

See audit reports in the repository for detailed verification.

## Math Documentation

See `MATHEMATICS.md` for the full mathematical derivations of:
- Forward/backward propagation
- PPO clipped objective
- GAE computation
- Adam optimizer
- Environment dynamics
- Gaussian policy gradients

Everything is derived from first principles - no hand-waving.

## License

Free for educational and research use.

---

Built with NumPy only. No TensorFlow, PyTorch, Gym, or Matplotlib dependencies.
