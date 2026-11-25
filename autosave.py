"""Periodic persistence for the Q-learning agent."""


class AutoSaver:
    """Save the agent's Q-table every fixed number of completed episodes."""

    def __init__(self, interval=100):
        if interval <= 0:
            raise ValueError("Autosave interval must be greater than zero.")
        self.interval = interval
        self.last_saved_episode = 0
        self.next_save_episode = interval

    def mark_saved(self, completed_episodes):
        """Record a manual or automatic save without causing a duplicate save.

        A manual save before the next scheduled boundary does not change the
        schedule. Saving at or beyond a boundary advances it to the next one.
        """
        self.last_saved_episode = completed_episodes
        if completed_episodes >= self.next_save_episode:
            self.next_save_episode = (
                completed_episodes // self.interval + 1
            ) * self.interval

    def save_if_due(self, agent, filename, completed_episodes):
        """Save at episode 100, 200, 300, etc.; return True when saved."""
        if completed_episodes < self.next_save_episode:
            return False

        agent.save(filename)
        self.mark_saved(completed_episodes)
        return True
