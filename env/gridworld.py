import numpy as np
import matplotlib.pyplot as plt

ACTIONS = ["N", "E", "S", "W"]
ACTION_TO_DELTA = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}


class Gridworld:
    def __init__(self, K=12, start=(1, 1), goal=None, traps=None):
        self.K = K
        self.start = start
        self.goal = goal if goal is not None else (K, K)
        self.traps = set(traps) if traps is not None else set()

    def get_states(self):
        states = []
        for x in range(1, self.K + 1):
            for y in range(1, self.K + 1):
                states.append((x, y))
        return states

    def is_terminal(self, state):
        return state == self.goal or state in self.traps

    def get_actions(self, state):
        if self.is_terminal(state):
            return []
        x, y = state
        actions = []
        if y < self.K:
            actions.append("N")
        if x < self.K:
            actions.append("E")
        if y > 1:
            actions.append("S")
        if x > 1:
            actions.append("W")
        return actions

    def step(self, state, action):
        if self.is_terminal(state):
            return state, 0.0, True

        dx, dy = ACTION_TO_DELTA[action]
        x, y = state
        nx = min(max(x + dx, 1), self.K)
        ny = min(max(y + dy, 1), self.K)
        next_state = (nx, ny)

        if next_state == self.goal:
            reward = 2 * (self.K - 1)
            done = True
        elif next_state in self.traps:
            reward = -2 * (self.K - 1)
            done = True
        else:
            reward = -1.0
            done = False

        return next_state, reward, done

    # --- plotting helpers ---

    def _state_to_indices(self, state):
        # map (x,y) with 1..K to indices 0..K-1
        x, y = state
        return y - 1, x - 1  # row, col

    def plot_values(self, V, title="Value function"):
        grid = np.zeros((self.K, self.K))
        for s in self.get_states():
            r, c = self._state_to_indices(s)
            grid[r, c] = V.get(s, 0.0)
        plt.figure()
        plt.imshow(grid, origin="lower", cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def plot_policy(self, policy, title="Policy"):
        plt.figure()
        for s in self.get_states():
            if self.is_terminal(s):
                continue
            actions = policy.get(s)
            if actions is None:
                continue
            x, y = s
            if isinstance(actions, list):
                for a in actions:
                    self._draw_arrow(x, y, a, color="k")
            else:
                self._draw_arrow(x, y, actions, color="k")

        # mark start, goal, traps
        sx, sy = self.start
        gx, gy = self.goal
        plt.scatter([sx], [sy], c="green", marker="s", label="Start")
        plt.scatter([gx], [gy], c="red", marker="*", label="Goal")
        if self.traps:
            tx = [t[0] for t in self.traps]
            ty = [t[1] for t in self.traps]
            plt.scatter(tx, ty, c="black", marker="x", label="Traps")

        plt.xlim(0.5, self.K + 0.5)
        plt.ylim(0.5, self.K + 0.5)
        plt.xticks(range(1, self.K + 1))
        plt.yticks(range(1, self.K + 1))
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.title(title)
        plt.legend()
        plt.show()

    def _draw_arrow(self, x, y, action, color="k"):
        dx, dy = ACTION_TO_DELTA[action]
        plt.arrow(x, y, 0.3 * dx, 0.3 * dy,
                  head_width=0.15, head_length=0.15, fc=color, ec=color)


def make_gridworld_no_traps(K=12, start=(1, 1), goal=None):
    return Gridworld(K=K, start=start, goal=goal)


def make_gridworld_with_traps(K=12, start=(1, 1), goal=None):
    if goal is None:
        goal = (K, K)
    traps = set()
    # U = {(x, y) | 1 <= x <= 8, y = 4} and {(x, y) | 5 <= x <= 12, y = 8}
    for x in range(1, 9):
        traps.add((x, 4))
    for x in range(5, 13):
        traps.add((x, 8))
    return Gridworld(K=K, start=start, goal=goal, traps=traps)
