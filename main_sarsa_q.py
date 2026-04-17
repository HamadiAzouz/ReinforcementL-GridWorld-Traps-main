from env.gridworld import make_gridworld_no_traps, make_gridworld_with_traps
from algos.sarsa import sarsa
from algos.q_learning import q_learning


def run_td(env, label="no_traps"):
    print(f"SARSA ({label})...")
    Q_sarsa, pi_sarsa = sarsa(env, num_episodes=5000, alpha=0.1,
                              gamma=0.99, epsilon=0.1)
    env.plot_policy(pi_sarsa, title=f"SARSA policy ({label})")

    print(f"Q-Learning ({label})...")
    Q_q, pi_q = q_learning(env, num_episodes=5000, alpha=0.1,
                           gamma=0.99, epsilon=0.1)
    env.plot_policy(pi_q, title=f"Q-learning policy ({label})")


if __name__ == "__main__":
    K = 12
    start = (1, 1)
    goal = (K, K)

    env_no = make_gridworld_no_traps(K=K, start=start, goal=goal)
    env_traps = make_gridworld_with_traps(K=K, start=start, goal=goal)

    run_td(env_no, label="no_traps")
    run_td(env_traps, label="with_traps")
