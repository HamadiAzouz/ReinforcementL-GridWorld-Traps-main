import numpy as np


def epsilon_greedy_from_Q(Q, state, env, epsilon):
    actions = env.get_actions(state)
    if not actions:
        return None
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    q_values = [Q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
    return np.random.choice(best_actions)


def generate_episode(env, policy_fn):
    s = env.start
    episode = []
    done = False
    while not done:
        a = policy_fn(s, env)
        if a is None:
            break
        s_next, r, done = env.step(s, a)
        episode.append((s, a, r))
        s = s_next
    return episode


def mc_policy_iteration(env, num_episodes=5000, gamma=0.99,
                        epsilon=0.1, first_visit=True):
    Q = {}
    returns_sum = {}
    returns_count = {}

    def policy_fn(state, env):
        return epsilon_greedy_from_Q(Q, state, env, epsilon)

    for ep in range(num_episodes):
        episode = generate_episode(env, policy_fn)
        # compute returns backwards
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if first_visit:
                if (s, a) in visited:
                    continue
                visited.add((s, a))
            # every-visit or first-visit update
            returns_sum[(s, a)] = returns_sum.get((s, a), 0.0) + G
            returns_count[(s, a)] = returns_count.get((s, a), 0) + 1
            Q[(s, a)] = returns_sum[(s, a)] / returns_count[(s, a)]

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
