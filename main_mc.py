from env.gridworld import make_gridworld_no_traps, make_gridworld_with_traps
from algos.monte_carlo import mc_policy_iteration


def run_mc(env, label="no_traps"):
    print(f"MC first-visit ({label})...")
    Q_fv, pi_fv = mc_policy_iteration(env, num_episodes=5000,
                                      gamma=0.99, epsilon=0.1,
                                      first_visit=True)
    env.plot_policy(pi_fv, title=f"MC first-visit policy ({label})")

    print(f"MC every-visit ({label})...")
    Q_ev, pi_ev = mc_policy_iteration(env, num_episodes=5000,
                                      gamma=0.99, epsilon=0.1,
                                      first_visit=False)
    env.plot_policy(pi_ev, title=f"MC every-visit policy ({label})")


if __name__ == "__main__":
    K = 12
    start = (1, 1)
    goal = (K, K)

    env_no = make_gridworld_no_traps(K=K, start=start, goal=goal)
    env_traps = make_gridworld_with_traps(K=K, start=start, goal=goal)

    run_mc(env_no, label="no_traps")
    run_mc(env_traps, label="with_traps")
