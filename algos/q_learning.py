import numpy as np


def epsilon_greedy(Q, state, env, epsilon):
    actions = env.get_actions(state)
    if not actions:
        return None
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    q_values = [Q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
    return np.random.choice(best_actions)


def q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}
    for ep in range(num_episodes):
        s = env.start
        done = False
        while not done:
            a = epsilon_greedy(Q, s, env, epsilon)
            s_next, r, done = env.step(s, a)
            q_sa = Q.get((s, a), 0.0)
            if done:
                target = r
            else:
                actions_next = env.get_actions(s_next)
                if actions_next:
                    max_q_next = max(Q.get((s_next, a2), 0.0) for a2 in actions_next)
                else:
                    max_q_next = 0.0
                target = r + gamma * max_q_next
            Q[(s, a)] = q_sa + alpha * (target - q_sa)
            s = s_next
    # greedy policy
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
