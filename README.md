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

---

## Results with Visualizations

### Dynamic Programming Results

#### **Figure 1: DP Value Function - No Traps (Value Iteration)**

![DP Value Function - No Traps](dp_plots/valuefunction_VI_notraps.png)

**Interpretation:**
- **Heatmap Pattern**: Shows a gradient of values decreasing from the top-right (goal at (12,12)) toward the bottom-left (start at (1,1))
- **Peak (Red/Yellow)**: Value of ~22 at the goal state - this is the maximum possible return (approximately 2(K-1) = 22)
- **Valley (Dark Blue)**: Values near 0 at the starting corner, reflecting the cost of reaching the goal
- **Color Gradient**: Each step away from the goal decreases value by approximately 1.0, corresponding to the -1 step cost
- **Optimality**: This is the **guaranteed optimal value function** - every cell's value represents the true maximum expected return following the optimal policy
- **No Obstacles**: The smooth linear gradient indicates no obstacles disrupt the shortest path
- **Algorithm**: Computed using **Value Iteration** - direct computation of optimal values via Bellman optimality equation

#### **Figure 2: DP Policy - No Traps (Value Iteration)**

![DP Policy - No Traps](dp_plots/policy_VI_notraps.png)

**Interpretation:**
- **Arrow Direction**: All arrows point toward the goal (northeast direction)
- **Systematic Pattern**: Arrows form a coherent "flow field" - the agent can follow arrows from any position and reach the goal
- **Shortest Paths**: Policies generally move diagonally (East + North simultaneously not possible, so alternates between E and N)
- **Optimal Navigation**: This is the **greedy policy extracted from the optimal value function** - guaranteed to find the shortest path
- **Green Square (1,1)**: Starting position
- **Red Star (12,12)**: Goal position
- **Deterministic**: Each state has exactly one best action (ties broken arbitrarily)
- **Algorithm**: Extracted from **Value Iteration's** optimal value function using greedy action selection

#### **Figure 3: DP Value Function - With Traps (Value Iteration)**

![DP Value Function - With Traps](dp_plots/valuefunction_VI_withtraps.png)

**Interpretation:**
- **Trap Regions (Dark Blue/Black)**: Large valleys appear at rows 4 and 8 where traps are located
- **Value Drop**: The two horizontal trap lines at y=4 and y=8 show as significant depressions in the value surface
- **Avoidance Cost**: States adjacent to traps have lower values because paths avoiding them are longer
- **Optimal Path**: The algorithm has "learned" (computed) the expensive detours needed to avoid traps
- **Asymmetry**: Values are lower on the right side because paths must navigate around the trap at (5,8) to (12,8)
- **Global Optimum**: Despite complexity, this is still the true optimal value function - the algorithm knows the best way to navigate the dangerous regions
- **Algorithm**: **Value Iteration** provides complete knowledge-based optimal solution accounting for all trap penalties

#### **Figure 4: DP Policy - With Traps (Value Iteration)**

![DP Policy - With Traps](dp_plots/policy_VI_withtraps.png)

**Interpretation:**
- **Trap Avoidance**: Arrows systematically steer around the trap regions (black X marks)
- **Detour Planning**: 
  - Below trap at y=4: Arrows route either left (West) then up, or up early then right
  - Below trap at y=8: Arrows route around left side, never directly through the trap line
- **Convergent Flow**: Multiple paths merge toward the goal, showing intelligent navigation around obstacles
- **Perfect Knowledge**: Being model-based, **Value Iteration** finds the provably optimal detours
- **Markers**:
  - **Green Square (1,1)**: Start
  - **Red Star (12,12)**: Goal
  - **Black X**: Traps - no arrows here (terminal states)
- **Robustness**: Regardless of starting position, following the arrows guarantees reaching the goal while avoiding all traps

---

#### **Figure 5: DP Value Function - No Traps (Policy Iteration)**

![DP Value Function - No Traps](dp_plots/valuefunctions_PI_notraps.png)

**Interpretation:**
- **Policy Iteration Convergence**: This value function is computed by **Policy Iteration**, an alternative to Value Iteration that alternates between policy evaluation and policy improvement
- **Identical to Value Iteration**: Should be virtually identical to Figure 1, showing the same smooth gradient from goal (peak ~22) to start (valley ~0)
- **Convergence Comparison**: Policy Iteration often converges in fewer outer iterations (3-10 policy improvement steps) but each iteration is more expensive (full policy evaluation required)
- **Theoretical Equivalence**: Both algorithms compute the same optimal value function $V^*(s)$, just via different computational paths
- **Practical Difference**: While the final values are the same, the convergence speed may differ—Policy Iteration's fewer outer loops vs. Value Iteration's many small sweeps
- **No Traps Simplicity**: The no-traps environment is so simple that both algorithms converge nearly instantly, making the difference negligible
- **Algorithm**: Computed using **Policy Iteration** - alternating evaluation and improvement until convergence

#### **Figure 6: DP Policy - No Traps (Policy Iteration)**

![DP Policy - No Traps](dp_plots/policy_PI_notraps.png)

**Interpretation:**
- **Policy Iteration Greedy Policy**: Extracted from **Policy Iteration's** learned value function using $\pi(s) = \arg\max_a [r + \gamma V(s')]$
- **Expected Identity**: Should be identical or nearly identical to Figure 2—both algorithms produce the same optimal policy arrows pointing northeast to goal
- **Deterministic Navigation**: Each state has exactly one best action; the policy is fully determined by the optimal value function
- **Shortest Path Navigation**: Like Value Iteration, Policy Iteration's policy navigates directly from (1,1) to (12,12) in the minimum number of steps
- **No Algorithmic Difference in Policy**: The extraction process is identical for both DP variants; differences in convergence speed don't affect the final policy
- **Convergence Mechanism**: **Policy Iteration** continues improving until the policy stabilizes—when no further improvements are possible, convergence is achieved

#### **Figure 7: DP Value Function - With Traps (Policy Iteration)**

![DP Value Function - With Traps](dp_plots/valuefunction_PI_withraps.png)

**Interpretation:**
- **Trap-Aware Value Function**: **Policy Iteration's** value function with full knowledge of trap locations at y=4 and y=8
- **Identical Optimal Values**: Should match Figure 3 exactly—both algorithms compute $V^*(s)$, the true optimal value accounting for expensive detours around traps
- **Value Landscape**: Shows clear valleys (dark blue) at trap regions and asymmetric terrain on the right side where the trap at (5,8) to (12,8) forces longer paths
- **Convergence Efficiency**: Policy Iteration may find this optimal solution with fewer outer iterations:
  - Early iterations: Rough value estimates and improving policy
  - Later iterations: Policy stabilizes, but full evaluation ensures correct values
- **Optimality Guarantee**: Like all DP methods with complete information, guaranteed to find the true optimal solution regardless of initial policy
- **Algorithm**: **Policy Iteration** provides optimal trap-aware navigation through alternating evaluation/improvement cycles

#### **Figure 8: DP Policy - With Traps (Policy Iteration)**

![DP Policy - With Traps](dp_plots/policy_PI_withtraps.png)

**Interpretation:**
- **Policy Iteration's Optimal Trap Avoidance**: The greedy policy extracted from **Policy Iteration's** optimal value function
- **Identical to Value Iteration**: Should be the same as Figure 4—both algorithms extract identical greedy policies from the optimal value function
- **Systematic Detour Routes**: 
  - Arrows below y=4 trap: Route around the obstacle via West or early North movement
  - Arrows below y=8 trap: Mostly route left, circling the trap region from the west
- **Perfect Navigation**: Following these arrows from any starting position guarantees reaching the goal while avoiding all traps
- **Policy Convergence**: In **Policy Iteration**, this policy emerges after the inner loop (policy evaluation) converges—the agent "commits" to actions that are greedy w.r.t. their true values
- **Convergence Mechanism**: Policy Iteration continues outer iterations until the policy stabilizes—once extracted policy equals input policy, the algorithm terminates with optimal policy
- **Optimal Safety**: The trap-avoidance behavior is provably optimal, computed through systematic evaluation-improvement loops

---

### Policy Iteration vs. Value Iteration Summary

Both DP variants guarantee optimal solutions:

| Aspect | Value Iteration | Policy Iteration |
|--------|---|---|
| **Update Type** | Direct Bellman optimality | Alternating eval/improvement |
| **Convergence** | Many value sweeps | Fewer outer loops + full evaluations |
| **Value Accuracy** | Improves incrementally | Exact per policy evaluation step |
| **Final Result** | Identical optimal values | Identical optimal values |
| **Final Policy** | Identical greedy policy | Identical greedy policy |
| **Computational Cost** | Many small updates | Fewer large updates |
| **Best For** | Fast implementation | Faster outer convergence |
| **File Names** | `valuefunction_VI_*`, `policy_VI_*` | `valuefunctions_PI_*`, `policy_PI_*` |

---

### Monte Carlo Results

#### **Figure 1: MC First-Visit Policy - No Traps**

![MC First-Visit Policy - No Traps](mc_plots/Figure_1.png)

**Interpretation:**
- **Suboptimal/Noisy Policy**: Unlike DP's clean policies, MC produces highly suboptimal, noisy navigation patterns even in the simple no-traps case
- **Why MC Struggles Here**:
  - **High Variance**: Monte Carlo's returns $G_t$ are unbiased but have massive variance. Gridworld episodes are long (up to 22+ steps), and random exploration means high variance in cumulative rewards
  - **Slow Convergence**: Each state-action pair needs many visits to get reliable value estimates. In a 12×12 grid with 4 actions, there are 576 state-action pairs—5000 episodes spread thin across all of them
  - **Exploration Overhead**: 10% random actions waste 500+ episodes on random noise, providing little learning signal
  - **Long Episodes**: Gridworld episodes average 15-22 steps. MC only updates after full episodes, so learning feedback is delayed and aggregated across many steps
- **Observable Defects**:
  - Arrows point in inconsistent directions even in similar states
  - Some states may have suboptimal actions selected
  - Policy is unstable and varies with random seeds
- **Fundamental Limitation**: Monte Carlo is fundamentally ill-suited for gridworld because the environment has long episodes and high variance rewards, both of which MC handles poorly

#### **Figure 2: MC Every-Visit Policy - No Traps**

![MC Every-Visit Policy - No Traps](mc_plots/Figure_2.png)

**Interpretation:**
- **Similar Suboptimal Behavior**: Every-visit MC produces similarly poor results as first-visit, confirming this is not a first-visit vs. every-visit issue
- **No Improvement Over First-Visit**: Despite using more data (every visit instead of first visit), the policy remains noisy and suboptimal
- **Why Every-Visit Doesn't Help**:
  - Still limited by high variance (cumulative returns are inherently noisy in long episodes)
  - Episode length variance dominates: longer episodes have noisier returns, shorter episodes have less data
  - More visits per episode doesn't resolve the fundamental variance problem—it just increases the variance in the value updates
- **Persistent Noise**: Arrows remain inconsistent and suboptimal even with double the updates per episode
- **Comparison**: Every-visit should theoretically provide more data per episode, but in gridworld's high-variance environment, this actually increases noise rather than improving learning
- **Key Insight**: The problem is not algorithmic choice (first vs. every-visit) but MC's fundamental unsuitability for this environment

#### **Figure 3: MC First-Visit Policy - With Traps**

![MC First-Visit Policy - With Traps](mc_plots/Figure_3.png)

**Interpretation:**
- **Complete Failure**: Even after 5000 episodes (and empirically tested with 25k, 50k, 100k episodes), MC fails to discover or properly avoid traps
- **Why MC Fails Catastrophically With Traps**:
  - **Rare Trap Encounters**: Traps occupy only 21 out of 144 cells (~15%). With ε=0.1 exploration, the agent rarely hits traps by chance, so it learns almost nothing about them
  - **Extreme Variance from Trap Penalties**: When the agent does hit a trap (reward -24), this creates an extremely negative return $G_t$. This huge swing in the return distribution makes learning unreliable—one lucky run vs. one unlucky trap hit creates massive value estimate swings
  - **Episode Length Volatility**: 
    - Normal episode to goal: ~11-15 steps, return ~$22 - 11 = 11$
    - Episode hitting early trap: 1-5 steps, return ~$-24 - (1 \text{ to } 5) = -25$ to $-29$
    - This enormous variance in returns makes value estimates unstable
  - **Insufficient Episodes**: Even 100,000 episodes cannot overcome the variance. With 4 actions × 144 states = 576 state-action pairs, and sparse trap encounters, many state-action pairs near traps are visited too rarely to build reliable estimates
  - **Parameter Tuning Doesn't Help**:
    - Increasing episodes to 25k, 50k, 100k: The variance problem doesn't go away, it just spreads the noisy samples across more updates
    - Changing ε to higher values (e.g., 0.2, 0.3): More exploration hits more traps, but also increases episode volatility and makes the learning signal noisier
    - Changing ε to lower values: Fewer trap hits means even slower learning about trap locations
- **Observable Failure Modes**:
  - Policy shows no consistent trap avoidance pattern
  - Arrows may point directly into trap regions
  - Navigation is random or suboptimal
  - Some states adjacent to traps still suggest moving toward them
- **Fundamental Unsuitability**: MC's dependence on full-episode returns makes it catastrophically bad for sparse, high-penalty events like traps in gridworld

#### **Figure 4: MC Every-Visit Policy - With Traps**

![MC Every-Visit Policy - With Traps](mc_plots/Figure_4.png)

**Interpretation:**
- **Persistent Failure**: Like first-visit, every-visit MC completely fails to learn proper trap avoidance despite 5000+ episodes and parameter tuning
- **Why Every-Visit Fails Even Worse**:
  - **No Data Reuse Benefit**: In gridworld, revisiting a state multiple times within an episode is rare. Most episodes take a roughly monotonic path toward the goal or directly hit a trap
  - **Increased Noise**: If an episode does revisit states, those visits are typically near the end (close to goal or trap), creating highly correlated, non-stationary data that amplifies variance
  - **Same Variance Problem**: Every-visit still uses full-episode returns, inheriting all the variance issues from first-visit MC
- **Empirically Confirmed Failure**:
  - Tested with 25k, 50k, 100k episodes: No improvement
  - Tested with ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5}: All fail to discover trap avoidance
  - Increasing episodes just makes the noisy learning happen more times—it doesn't solve the fundamental problem
- **Why Tuning Doesn't Work**:
  - More episodes ≠ better learning when samples are high-variance
  - Higher ε ≠ more trap discovery; it just increases all variance
  - Lower ε ≠ faster convergence; it starves the agent of feedback
- **Conclusion**: Monte Carlo is fundamentally unsuitable for gridworld-like environments with sparse, high-penalty events. The algorithm cannot overcome the variance inherent in long episodes and rare trap encounters, regardless of episode count or exploration rate

---

### Temporal Difference Results

#### **Figure 1: SARSA Policy - No Traps**

![SARSA Policy - No Traps](sarsa_q_plots/Figure_1.png)

**Interpretation:**
- **On-Policy Learning**: SARSA learns the value of its own exploratory policy (epsilon-greedy with ε=0.1)
- **Conservative Navigation**: Since 10% of actions are random, SARSA learns values that account for this exploration cost
- **Fast Learning**: SARSA learns much faster than MC (typically converges in 100-500 episodes) due to bootstrapping
- **Policy Quality**: Arrows point toward goal, though may show some exploration-induced suboptimality
- **Stable Convergence**: SARSA is very stable and rarely diverges, making it safe for online learning
- **Practical Speed**: 5000 episodes is more than sufficient; convergence likely happened by episode 1000

#### **Figure 2: Q-Learning Policy - No Traps**

![Q-Learning Policy - No Traps](sarsa_q_plots/Figure_2.png)

**Interpretation:**
- **Off-Policy Learning**: Q-learning learns the optimal policy while exploring (epsilon-greedy ε=0.1)
- **Aggressive Optimization**: Q-learning uses max(Q(s',a')) in updates, always learning toward the best action
- **Faster Convergence**: Q-learning typically converges faster than SARSA (fewer episodes needed)
- **Optimal Aspiration**: Q-values estimate optimal returns, even though the agent explores with ε=0.1
- **Possible Overestimation**: Early in learning, Q-values may be overestimated (maximization bias), but this diminishes with more episodes
- **Superior Quality**: In the no-traps case, Q-learning should produce the most consistently optimal policy

#### **Figure 3: SARSA Policy - With Traps**

![SARSA Policy - With Traps](sarsa_q_plots/Figure_3.png)

**Interpretation:**
- **Learned Avoidance**: Discovered trap locations through experience and learned to avoid them
- **Conservative Routes**: SARSA's on-policy nature makes it extra cautious:
  - Accounts for the 10% probability of random exploration
  - May take safer, longer paths than Q-learning to avoid risk
  - Avoidance behavior is robust to the exploration policy
- **Stable Learning**: Being conservative, SARSA rarely falls into traps after initial learning
- **Safe for Real Systems**: This caution is valuable in robotics/autonomous systems where mistakes are costly
- **Computational Efficiency**: Still converges quickly (bootstrapping advantage over MC)
- **Quality Tradeoff**: Path may be longer than DP's optimal, but learned safely from experience

#### **Figure 4: Q-Learning Policy - With Traps**

![Q-Learning Policy - With Traps](sarsa_q_plots/Figure_4.png)

**Interpretation:**
- **Aggressive Trap Avoidance**: Q-learning learned to avoid traps while optimizing for shortest paths
- **Near-Optimal Navigation**: Off-policy learning allows Q-learning to discover near-optimal detours
- **Convergence to Optimality**: Given enough episodes (5000 is sufficient), Q-learning approximates the DP solution
- **Aggressive Routes**: Compared to SARSA, may take more aggressive (shorter) paths since it learns optimal values regardless of exploration
- **Risk During Learning**: During training, Q-learning may explore traps more (higher TD errors), but converges faster overall
- **Practical Efficiency**: The off-policy nature makes Q-learning sample-efficient and widely used in practice
- **Strong Performance**: Should match or slightly exceed SARSA's final policy quality while learning faster

---

## Algorithm Comparison

### Convergence Speed (Episodes to Reasonable Performance)

| Algorithm | No Traps | With Traps | Notes |
|-----------|----------|-----------|-------|
| **DP (Value Iteration)** | ~10-20 iterations | ~10-20 iterations | Model-based, instant if model known |
| **Q-Learning** | 50-200 | 200-500 | Fastest model-free (off-policy) |
| **SARSA** | 100-500 | 500-2000 | Faster than MC but slower than Q |
| **Monte Carlo** | 1000-5000 | 2000-5000 | Slowest (requires full episodes) |

### Policy Quality After 5000 Episodes

| Algorithm | No Traps | With Traps | Optimality |
|-----------|----------|-----------|-----------|
| **DP** | Perfect | Perfect | Guaranteed optimal |
| **Q-Learning** | Near-optimal | Near-optimal | Converges to optimal |
| **SARSA** | Near-optimal | Near-optimal | Converges to ε-optimal |
| **Monte Carlo** | Good | Functional | Converges to optimal |

### Sample Efficiency (Utilization of Experience)

1. **Q-Learning**: Highest (learns from all exploratory actions)
2. **SARSA**: Medium (learns from on-policy transitions)
3. **Monte Carlo**: Lower (aggregates entire episodes)
4. **DP**: N/A (doesn't need samples)

### Risk During Learning

1. **SARSA**: Lowest risk (conservative, accounts for exploration)
2. **Monte Carlo**: Low risk (learns from safe rollouts)
3. **Q-Learning**: Moderate risk (explores aggressively, overestimates)
4. **DP**: No risk (just computation)

### Practical Applicability

**Best for Unknown Environments**: Q-Learning and SARSA
- Both learn from experience without knowing the model
- Q-Learning is faster; SARSA is safer

**Best for Safety-Critical Tasks**: SARSA and Monte Carlo
- SARSA: Accounts for exploration in value estimates
- MC: Conservative, learns from complete trajectories

**Best When Model Available**: Dynamic Programming
- Guaranteed optimal solutions
- Instant convergence (within computational limits)

---

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
