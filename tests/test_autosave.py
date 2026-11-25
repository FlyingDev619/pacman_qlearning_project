import tempfile
import unittest
from pathlib import Path

from agent import QLearningAgent
from autosave import AutoSaver


class AutoSaverTests(unittest.TestCase):
    def test_saves_only_at_100_completed_episode_intervals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            qtable_path = Path(temp_dir) / "q_table.json"
            agent = QLearningAgent(seed=10)
            agent.q_table["sample-state"] = [1.0, 2.0, 3.0, 4.0]
            saver = AutoSaver(interval=100)

            # Game steps must not drive this counter: only completed episodes do.
            self.assertFalse(saver.save_if_due(agent, qtable_path, 99))
            self.assertFalse(qtable_path.exists())

            self.assertTrue(saver.save_if_due(agent, qtable_path, 100))
            self.assertEqual(saver.last_saved_episode, 100)
            self.assertEqual(saver.next_save_episode, 200)

            loaded_agent = QLearningAgent()
            loaded_agent.load(qtable_path)
            self.assertEqual(
                loaded_agent.q_table["sample-state"], [1.0, 2.0, 3.0, 4.0]
            )

            self.assertFalse(saver.save_if_due(agent, qtable_path, 199))
            self.assertTrue(saver.save_if_due(agent, qtable_path, 200))
            self.assertEqual(saver.last_saved_episode, 200)
            self.assertEqual(saver.next_save_episode, 300)
            self.assertFalse(saver.save_if_due(agent, qtable_path, 200))

    def test_manual_save_before_boundary_keeps_schedule(self):
        saver = AutoSaver(interval=100)
        saver.mark_saved(45)
        self.assertEqual(saver.last_saved_episode, 45)
        self.assertEqual(saver.next_save_episode, 100)

    def test_manual_save_at_boundary_advances_schedule(self):
        saver = AutoSaver(interval=100)
        saver.mark_saved(100)
        self.assertEqual(saver.last_saved_episode, 100)
        self.assertEqual(saver.next_save_episode, 200)

    def test_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            AutoSaver(interval=0)


if __name__ == "__main__":
    unittest.main()
