from env.gridworld import make_gridworld_no_traps, make_gridworld_with_traps
from algos.dp import value_iteration, policy_iteration


def run_dp(env, label="no_traps"):
    print(f"Running DP ({label})...")
    V_vi, pi_vi = value_iteration(env, gamma=0.99)
    V_pi, pi_pi = policy_iteration(env, gamma=0.99)

    env.plot_values(V_vi, title=f"Value function (VI, {label})")
    env.plot_policy(pi_vi, title=f"Policy (VI, {label})")

    env.plot_values(V_pi, title=f"Value function (PI, {label})")
    env.plot_policy(pi_pi, title=f"Policy (PI, {label})")


if __name__ == "__main__":
    K = 12
    start = (1, 1)
    goal = (K, K)

    env_no = make_gridworld_no_traps(K=K, start=start, goal=goal)
    env_traps = make_gridworld_with_traps(K=K, start=start, goal=goal)

    run_dp(env_no, label="no_traps")
    run_dp(env_traps, label="with_traps")
