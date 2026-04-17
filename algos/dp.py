import numpy as np


def value_iteration(env, gamma=0.99, theta=1e-4):
    states = env.get_states()
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        for s in states:
            if env.is_terminal(s):
                continue
            actions = env.get_actions(s)
            best = -np.inf
            for a in actions:
                s_next, r, done = env.step(s, a)
                val = r + gamma * V[s_next]
                if val > best:
                    best = val
            old = V[s]
            V[s] = best
            delta = max(delta, abs(old - V[s]))
        if delta < theta:
            break

    policy = {}
    for s in states:
        if env.is_terminal(s):
            continue
        actions = env.get_actions(s)
        best_a = None
        best = -np.inf
        for a in actions:
            s_next, r, done = env.step(s, a)
            val = r + gamma * V[s_next]
            if val > best:
                best = val
                best_a = a
        policy[s] = best_a
    return V, policy


def policy_evaluation(env, policy, gamma=0.99, theta=1e-4):
    states = env.get_states()
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        for s in states:
            if env.is_terminal(s):
                continue
            a = policy[s]
            s_next, r, done = env.step(s, a)
            v_new = r + gamma * V[s_next]
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V


def policy_iteration(env, gamma=0.99, theta=1e-4):
    states = env.get_states()
    policy = {}
    for s in states:
        if env.is_terminal(s):
            continue
        actions = env.get_actions(s)
        policy[s] = actions[0]

    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        stable = True
        for s in states:
            if env.is_terminal(s):
                continue
            old_a = policy[s]
            actions = env.get_actions(s)
            best_a = None
            best = -np.inf
            for a in actions:
                s_next, r, done = env.step(s, a)
                val = r + gamma * V[s_next]
                if val > best:
                    best = val
                    best_a = a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
        if stable:
            break
    return V, policy
