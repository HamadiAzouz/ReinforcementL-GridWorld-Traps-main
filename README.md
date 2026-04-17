# ReinforcementL-GridWorld-Traps-main
Gridworld Traps: Compare RL algorithms (Dynamic Programming, Monte Carlo, SARSA, Q-Learning) in a 12×12 grid with obstacles. Model-based &amp; model-free 
approaches, policy/value visualizations, flexible config. Educational tool for studying convergence, exploration, and trap avoidance in discrete MDPs. 🎯🤖
# Gridworld Traps: Reinforcement Learning Algorithms Comparison with Visualizations

A comprehensive implementation and comparison of multiple reinforcement learning algorithms applied to a gridworld environment with obstacles (traps). This project explores both model-based and model-free approaches to solve navigation problems in discrete state spaces, with detailed visualizations and interpretations.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Results with Visualizations](#results-with-visualizations)
  - [Dynamic Programming Results](#dynamic-programming-results)
  - [Monte Carlo Results](#monte-carlo-results)
  - [Temporal Difference Results](#temporal-difference-results)
- [Algorithm Comparison](#algorithm-comparison)
- [Configuration](#configuration)
- [File Descriptions](#file-descriptions)

## Overview

This project implements and compares four fundamental reinforcement learning algorithms to find optimal policies for navigating a gridworld environment. The environment includes two variants:
1. **No Traps**: A simple navigation task from start to goal
2. **With Traps**: A challenging navigation task that must avoid specific trap locations

### Key Features

- **Model-Based Methods**: Dynamic Programming algorithms that require knowledge of the environment
- **Model-Free Methods**: Monte Carlo and Temporal Difference learning that learn from experience
- **Visualization**: Policy and value function visualizations for all algorithms
- **Flexible Configuration**: Easy parameter tuning for different experimental setups
- **Scalable Grid**: Support for grids of various sizes (default 12×12)

## Project Structure

```
GridworldTraps/
├── README.md                 # Original README
├── main_dp.py               # Dynamic Programming entry point
├── main_mc.py               # Monte Carlo entry point
├── main_sarsa_q.py          # Temporal Difference entry point
│
├── env/
│   └── gridworld.py         # Gridworld environment implementation
│
├── algos/
│   ├── dp.py               # Value Iteration & Policy Iteration
│   ├── monte_carlo.py      # Monte Carlo Policy Iteration
│   ├── sarsa.py            # SARSA (On-Policy TD Learning)
│   └── q_learning.py       # Q-Learning (Off-Policy TD Learning)
│
├── dp_plots/                # DP visualizations
├── mc_plots/                # MC visualizations
└── sarsa_q_plots/           # TD visualizations
```

## Environment

### Gridworld Description

The gridworld is a discrete 2D environment with:

- **Grid Size**: K × K cells (default K = 12)
- **Coordinate System**: (x, y) with 1 ≤ x, y ≤ K
- **Start Position**: (1, 1) by default
- **Goal Position**: (K, K) by default
- **Actions**: 4 cardinal directions - North (N), East (E), South (S), West (W)
- **Boundary Behavior**: Movement is clipped at grid boundaries

### Reward Structure

- **Goal State**: +24 reward (2 × (K - 1) with K = 12)
- **Trap State**: -24 reward (-2 × (K - 1))
- **Step Cost**: -1 reward per step (encourages shorter paths)
- **Terminal States**: Goal and trap states are terminal (episode ends)

### Traps Configuration

In the "with_traps" variant, traps are placed at:
- **Row 4**: All cells (1, 4) to (8, 4) - horizontal obstacle
- **Row 8**: All cells (5, 8) to (12, 8) - horizontal obstacle

These traps create challenging navigation scenarios that require planning to avoid.

## Algorithms

### 1. Dynamic Programming (DP)

**Type**: Model-Based, Planning

**Variants Implemented**:

#### Value Iteration
- Iteratively improves value function until convergence
- Directly computes optimal value function
- Convergence Parameter: θ = 1e-4

#### Policy Iteration
- Alternates between policy evaluation and policy improvement
- Policy evaluation: Compute value function for current policy
- Policy improvement: Extract greedy policy from value function
- May converge faster than value iteration in some cases

**Advantages**:
- Guaranteed to find optimal policy (given complete environment model)
- Deterministic results
- Theoretically well-understood

**Disadvantages**:
- Requires complete knowledge of environment (transition probabilities and rewards)
- Can be computationally expensive for large state spaces

**File**: [algos/dp.py](algos/dp.py)

### 2. Monte Carlo (MC) Policy Iteration

**Type**: Model-Free, Learning from Experience

**Variants Implemented**:

#### First-Visit Monte Carlo
- Updates value estimates only on first visit to each state-action pair in an episode
- Provides unbiased estimates
- Lower variance than every-visit

#### Every-Visit Monte Carlo
- Updates value estimates on every visit to each state-action pair
- May be biased but converges to true value
- Can have higher variance

**Key Features**:
- Uses epsilon-greedy exploration (ε = 0.1)
- Learns from complete episodes
- Returns accumulated and averaged from all episodes

**Advantages**:
- Model-free (no need to know environment dynamics)
- Can learn from experience
- Works well in stochastic environments

**Disadvantages**:
- Requires complete episodes (poor for long-horizon tasks)
- Higher variance in value estimates
- Slower convergence compared to TD methods
- Cannot be applied to continuing tasks

**File**: [algos/monte_carlo.py](algos/monte_carlo.py)

### 3. SARSA (State-Action-Reward-State-Action)

**Type**: Model-Free, Temporal Difference (TD), On-Policy

**Key Characteristics**:
- Uses bootstrapping (updates based on one-step lookahead)
- On-policy: Learns value of policy being followed
- Updates: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

**Key Features**:
- Learning Rate (α): 0.1
- Discount Factor (γ): 0.99
- Exploration (ε): 0.1 (epsilon-greedy)
- Learns from real action transitions during training

**Advantages**:
- Faster learning than Monte Carlo (TD bootstrap)
- Works in continuing tasks
- Online learning capability
- Guaranteed convergence under certain conditions

**Disadvantages**:
- On-policy learning (wastes experience from exploratory actions)
- Can be sensitive to hyperparameter choices
- May not find optimal policy if exploration is limited

**File**: [algos/sarsa.py](algos/sarsa.py)

### 4. Q-Learning

**Type**: Model-Free, Temporal Difference (TD), Off-Policy

**Key Characteristics**:
- Uses bootstrapping with optimal action value
- Off-policy: Learns optimal policy while following exploratory policy
- Updates: Q(s,a) ← Q(s,a) + α[r + γ max(Q(s',·)) - Q(s,a)]

**Key Features**:
- Learning Rate (α): 0.1
- Discount Factor (γ): 0.99
- Exploration (ε): 0.1 (epsilon-greedy)
- Decouples exploration from learning

**Advantages**:
- Off-policy learning (can learn from any exploratory policy)
- Typically finds optimal policy faster than SARSA
- Works in continuing tasks
- Very practical and widely used

**Disadvantages**:
- Can overestimate values (maximization bias)
- May diverge in some conditions (though convergent for tabular case)
- Requires careful exploration-exploitation tradeoff

**File**: [algos/q_learning.py](algos/q_learning.py)

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib

### Setup

1. **Clone or download the project**:
```bash
cd TP2_GridworldTraps
```

2. **Install dependencies**:
```bash
pip install numpy matplotlib
```

Or using conda:
```bash
conda install numpy matplotlib
```

## Usage

### Running Dynamic Programming Algorithms

```bash
python main_dp.py
```

Generates visualizations for:
- Value functions (Value Iteration & Policy Iteration)
- Learned policies (Value Iteration & Policy Iteration)
- Both environments (no traps and with traps)

Output is saved to the `dp_plots/` directory.

### Running Monte Carlo Algorithms

```bash
python main_mc.py
```

Generates visualizations for:
- First-visit Monte Carlo policies
- Every-visit Monte Carlo policies
- Both environments (no traps and with traps)
- Requires 5000 episodes for convergence

Output is saved to the `mc_plots/` directory.

### Running Temporal Difference Algorithms

```bash
python main_sarsa_q.py
```

Generates visualizations for:
- SARSA learned policies
- Q-Learning learned policies
- Both environments (no traps and with traps)
- Requires 5000 episodes for convergence

Output is saved to the `sarsa_q_plots/` directory.

### Running All Algorithms

Execute scripts sequentially:
```bash
python main_dp.py && python main_mc.py && python main_sarsa_q.py
```

## Configuration

### Gridworld Parameters

Edit in main files to change:

```python
K = 12              # Grid size (K x K)
start = (1, 1)      # Starting position
goal = (K, K)       # Goal position
```

### Algorithm Parameters

Adjust in corresponding main files:

```python
# Monte Carlo
num_episodes = 5000  # Number of training episodes
gamma = 0.99         # Discount factor
epsilon = 0.1        # Exploration rate
first_visit = True   # True for first-visit, False for every-visit

# SARSA & Q-Learning
num_episodes = 5000  # Number of training episodes
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 0.1        # Exploration rate
```

### Convergence Parameters

Edit in [algos/dp.py](algos/dp.py):

```python
theta = 1e-4         # Value function convergence threshold
```

## File Descriptions

### Entry Points

- **[main_dp.py](main_dp.py)**: Runs Value Iteration and Policy Iteration, visualizes value functions and policies
- **[main_mc.py](main_mc.py)**: Runs First-Visit and Every-Visit Monte Carlo algorithms with 5000 episodes
- **[main_sarsa_q.py](main_sarsa_q.py)**: Runs SARSA and Q-Learning algorithms with 5000 episodes

### Core Modules

#### [env/gridworld.py](env/gridworld.py)
Implements the Gridworld environment:
- `Gridworld` class: Core environment with state/action/reward dynamics
- `make_gridworld_no_traps()`: Factory function for trap-free environment
- `make_gridworld_with_traps()`: Factory function for environment with predefined traps
- Visualization methods for policies and value functions

#### [algos/dp.py](algos/dp.py)
Dynamic Programming algorithms:
- `value_iteration()`: Computes optimal value function and policy
- `policy_iteration()`: Alternates between evaluation and improvement
- `policy_evaluation()`: Helper function for policy iteration

#### [algos/monte_carlo.py](algos/monte_carlo.py)
Monte Carlo learning:
- `mc_policy_iteration()`: Main algorithm supporting both first-visit and every-visit variants
- `generate_episode()`: Simulates an episode following a policy
- `epsilon_greedy_from_Q()`: Action selection using epsilon-greedy exploration

#### [algos/sarsa.py](algos/sarsa.py)
SARSA on-policy temporal difference learning:
- `sarsa()`: Main SARSA algorithm
- `epsilon_greedy()`: Epsilon-greedy action selection

#### [algos/q_learning.py](algos/q_learning.py)
Q-Learning off-policy temporal difference learning:
- `q_learning()`: Main Q-Learning algorithm
- `epsilon_greedy()`: Epsilon-greedy action selection

## Key Concepts

### Reinforcement Learning Terminology

- **State (s)**: A cell position in the gridworld
- **Action (a)**: A direction to move (N, E, S, W)
- **Reward (r)**: Numerical feedback (+24 at goal, -24 at trap, -1 per step)
- **Policy (π)**: Mapping from states to actions
- **Value Function V(s)**: Expected cumulative reward from state s
- **Action-Value Q(s,a)**: Expected cumulative reward for action a in state s
- **Return G**: Cumulative discounted reward from current step onward
- **Discount Factor γ**: Importance of future rewards (0.99 = 99%)

### Key Equations

**Bellman Expectation Equation**:
$$V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a)[r + \gamma V^{\pi}(s')]$$

**Bellman Optimality Equation**:
$$V^*(s) = \max_a \sum_{s',r} P(s',r|s,a)[r + \gamma V^*(s')]$$

**Q-Learning Update**:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**SARSA Update**:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$$

**Monte Carlo Return**:
$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$$

## Interpretation Guide

### Reading Value Function Heatmaps
- **Warmer Colors (Red/Yellow)**: Higher values (better states to be in)
- **Cooler Colors (Blue)**: Lower values (worse states to avoid)
- **Color Gradient**: Shows value landscape - steeper gradients mean bigger value differences between adjacent cells
- **Goal Region**: Always has peak values (~22 with K=12)
- **Trap Regions**: Show as valleys (low values) representing penalty and detour costs

### Reading Policy Arrows
- **Arrow Direction**: Indicates the action the policy takes at that state
- **Coherent Patterns**: Good policies show arrows forming "flow fields" that guide the agent toward goals
- **Avoidance**: Arrows systematically route around obstacles and traps
- **Uniformity**: Optimal policies often show uniform patterns (similar actions in similar situations)
- **Singularities**: Areas where arrows change direction abruptly may indicate state boundaries or cost discontinuities

### Comparing Algorithms Through Visualizations
- **Value Functions**: DP should be smoothest (optimal); others may show noise from sampling
- **Policies**: All should converge to similar paths; differences reflect learning inefficiency or exploration effects
- **Convergence**: No-trap version should show faster convergence for all algorithms
- **Robustness**: Policies learned in with-traps environment show algorithm's ability to handle obstacles

## Troubleshooting

### No visualizations appear

Ensure matplotlib is properly installed and configured:
```bash
pip install --upgrade matplotlib
```

### Memory issues with larger grids

If using K > 50, consider:
- Reducing episodes for MC/TD methods
- Using value iteration instead of policy iteration for DP
- Processing one environment at a time

### Convergence issues

If algorithms don't converge:
- Increase number of episodes (MC/TD)
- Reduce convergence threshold θ for DP
- Check exploration parameter ε is > 0 for MC/TD/SARSA

### Plot interpretation confusion

Refer to the [Interpretation Guide](#interpretation-guide) section above for detailed explanations of what colors, arrows, and patterns in the visualizations mean.

## Extensions and Improvements

Potential enhancements:

1. **Stochastic Environment**: Add randomness to action outcomes
2. **Function Approximation**: Use neural networks for large state spaces
3. **Continuous Action Space**: Extend to continuous control problems
4. **Multi-Agent**: Multiple agents learning simultaneously
5. **Imitation Learning**: Learn from expert demonstrations
6. **Performance Metrics**: Track cumulative reward and convergence speed during learning
7. **Hyperparameter Optimization**: Automated tuning of learning parameters
8. **Policy Visualization**: Create video demonstrations of learned policies
9. **Learning Curves**: Plot reward over episodes to show convergence

## References

### Key Papers

- Bellman, R. E. (1957). "Dynamic Programming"
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.)
- Watkins, C. J., & Dayan, P. (1992). "Q-learning"
- Rummery, G. A., & Niranjan, M. (1994). "On-Line Q-Learning Using Connectionist Systems"

### Textbooks

- Sutton & Barto: "Reinforcement Learning: An Introduction"
- Russell & Norvig: "Artificial Intelligence: A Modern Approach"
- Bertsekas & Tsitsiklis: "Neuro-Dynamic Programming"


**Last Updated**: January 2026

**Python Version**: 3.7+

**Status**: Complete and functional with comprehensive visualizations
