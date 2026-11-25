"""Tabular Q-learning agent with versioned state serialization."""

import json
import os
import random
import tempfile

from config import ACTIONS


STATE_VERSION = 4


class QLearningAgent:
    """Epsilon-greedy Q-learning over a compact strategic maze state."""

    def __init__(self, alpha=0.16, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.9985, min_epsilon=0.05, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}  # JSON-encoded state -> four action values
        self.read_only = False
        self.random = random.Random(seed)

    @staticmethod
    def encode_state(state):
        return json.dumps(state, separators=(",", ":"))

    def values_for(self, state):
        key = self.encode_state(state)
        if key not in self.q_table:
            if self.read_only:
                return [0.0] * len(ACTIONS)
            self.q_table[key] = [0.0] * len(ACTIONS)
        return self.q_table[key]

    def choose_action(self, state, legal_actions, explore=True, action_bias=None):
        """Choose an action using Q-values plus an optional model-based prior.

        The prior is calculated from the unchanged ghost transition rules. It
        guides early exploration toward moves with a safer two-turn outlook;
        learned Q-values break close ties and are still updated normally.
        """
        if not legal_actions:
            return None

        bias = action_bias or {}
        if explore and self.random.random() < self.epsilon:
            if bias:
                best_bias = max(bias.get(action, float("-inf"))
                                for action in legal_actions)
                guided = [action for action in legal_actions
                          if bias.get(action, float("-inf")) >= best_bias - 0.15]
                return self.random.choice(guided or legal_actions)
            return self.random.choice(legal_actions)

        values = self.values_for(state)
        if bias:
            # Normalize Q-values to a small tie-breaking term so the learned
            # table can refine, but not overturn, a strong immediate safety
            # estimate before it has seen enough successful trajectories.
            q_min = min(values[action] for action in legal_actions)
            q_max = max(values[action] for action in legal_actions)
            q_span = max(1.0, q_max - q_min)
            scores = {
                action: bias.get(action, float("-inf"))
                        + 2.0 * (values[action] - q_min) / q_span
                for action in legal_actions
            }
        else:
            scores = {action: values[action] for action in legal_actions}

        best_value = max(scores.values())
        best_actions = [action for action, value in scores.items()
                        if value == best_value]
        return self.random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_legal_actions,
              terminal=False):
        if self.read_only or self.alpha == 0.0:
            return
        current_values = self.values_for(state)
        old_value = current_values[action]

        if terminal or not next_legal_actions:
            future_value = 0.0
        else:
            next_values = self.values_for(next_state)
            future_value = max(next_values[index] for index in next_legal_actions)

        target = reward + self.gamma * future_value
        current_values[action] = old_value + self.alpha * (target - old_value)

    def end_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        data = {
            "state_version": STATE_VERSION,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "q_table": self.q_table,
        }
        directory = os.path.dirname(os.path.abspath(filename))
        fd, temporary = tempfile.mkstemp(prefix=".qtable-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, separators=(",", ":"))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def load(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        version = data.get("state_version", 1)
        if version != STATE_VERSION:
            raise ValueError(
                f"Q-table state version {version} is incompatible with "
                f"current state version {STATE_VERSION}; train a new table."
            )
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.min_epsilon = data.get("min_epsilon", self.min_epsilon)
        self.q_table = data.get("q_table", {})
