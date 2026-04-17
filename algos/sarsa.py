import numpy as np
from env.gridworld import ACTIONS


def epsilon_greedy(Q, state, env, epsilon):
    actions = env.get_actions(state)
    if not actions:
        return None
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    # greedy
    q_values = [Q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    # tie-breaking
    best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
    return np.random.choice(best_actions)


def sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}
    for ep in range(num_episodes):
        s = env.start
        a = epsilon_greedy(Q, s, env, epsilon)
        done = False
        while not done:
            s_next, r, done = env.step(s, a)
            a_next = epsilon_greedy(Q, s_next, env, epsilon)
            q_sa = Q.get((s, a), 0.0)
            target = r
            if not done and a_next is not None:
                target += gamma * Q.get((s_next, a_next), 0.0)
            Q[(s, a)] = q_sa + alpha * (target - q_sa)
            s, a = s_next, a_next
    # derive greedy policy
    policy = {}
    for s in env.get_states():
        if env.is_terminal(s):
            continue
        actions = env.get_actions(s)
        if not actions:
            continue
        q_values = [Q.get((s, a), 0.0) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        policy[s] = best_actions[0]
    return Q, policy
