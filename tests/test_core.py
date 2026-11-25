import unittest

from agent import QLearningAgent
from config import ACTIONS, AMBUSHER_START, CHASER_START, PACMAN_START
from game import Game
from ghost import Chaser
from maze import Maze
from pacman import Pacman


class MazeTests(unittest.TestCase):
    def test_maze_matches_the_requested_layout_exactly(self):
        expected = (
            "###############",
            "#.............#",
            "#.###.###.###.#",
            "#.........#...#",
            "#.#.#.###.#.#.#",
            "#...#...#...#.#",
            "#.###.#.#.###.#",
            "#.....#.#.....#",
            "#.###.#.#.###.#",
            "#.#...#.......#",
            "#.#.#.###.#.###",
            "#...#.........#",
            "#.###.###.###.#",
            "#.............#",
            "###############",
        )
        maze = Maze()
        self.assertEqual(maze.layout, expected)
        self.assertEqual(len(maze.floor), 111)
        self.assertEqual(len(maze.walls), 114)

    def test_maze_is_15_by_15_and_connected(self):
        maze = Maze()
        self.assertEqual((maze.width, maze.height), (15, 15))
        self.assertEqual(len(maze.floor) + len(maze.walls), 225)

        reachable = {PACMAN_START}
        frontier = [PACMAN_START]
        while frontier:
            current = frontier.pop()
            for neighbor in maze.legal_neighbors(current):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        self.assertEqual(reachable, maze.floor)

    def test_every_walkable_cell_starts_with_a_pellet(self):
        maze = Maze()
        self.assertEqual(maze.pellets, maze.floor)
        self.assertFalse(maze.walls & maze.pellets)

    def test_all_start_positions_are_walkable_and_distinct(self):
        maze = Maze()
        positions = {PACMAN_START, CHASER_START, AMBUSHER_START}
        self.assertEqual(len(positions), 3)
        self.assertTrue(all(maze.is_walkable(position) for position in positions))


class AgentTests(unittest.TestCase):
    def test_agent_chooses_only_a_legal_action(self):
        agent = QLearningAgent(seed=1)
        action = agent.choose_action((1, 2, 3), [0, 2])
        self.assertIn(action, [0, 2])

    def test_read_only_evaluation_does_not_decay_epsilon(self):
        agent = QLearningAgent(epsilon=0.0, epsilon_decay=0.5)
        agent.read_only = True
        agent.end_episode()
        self.assertEqual(agent.epsilon, 0.0)

    def test_learning_updates_a_q_value(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.0, epsilon=0.0)
        agent.learn((1,), 0, 2.0, (2,), [], terminal=True)
        self.assertAlmostEqual(agent.values_for((1,))[0], 1.0)

    def test_model_prior_guides_action_before_q_values_are_learned(self):
        agent = QLearningAgent(epsilon=0.0, seed=9)
        chosen = agent.choose_action((1,), [0, 1], action_bias={0: 0.0, 1: 3.0})
        self.assertEqual(chosen, 1)


class GhostTests(unittest.TestCase):
    def test_ghost_speed_is_seven_moves_per_ten_player_turns(self):
        ghost = Chaser((13, 1))
        moves = sum(1 for _ in range(10) if ghost.should_move_this_turn())
        self.assertEqual(moves, 7)

    def test_ambusher_predicts_three_tiles_ahead_in_open_corridor(self):
        maze = Maze()
        pacman = Pacman((1, 1))
        pacman.direction = "RIGHT"
        self.assertEqual(pacman.predict_position(maze, steps=3), (4, 1))


class GameTests(unittest.TestCase):
    def test_game_starts_with_one_life_and_expected_entities(self):
        game = Game(QLearningAgent(seed=2))
        self.assertFalse(game.terminal)
        self.assertEqual(game.pacman.position, PACMAN_START)
        self.assertEqual(game.chaser.position, CHASER_START)
        self.assertEqual(game.ambusher.position, AMBUSHER_START)
        self.assertEqual(len(game.ghosts), 2)

    def test_state_is_a_hashable_tuple(self):
        game = Game(QLearningAgent(seed=3))
        state = game.observe()
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 18)
        self.assertTrue(all(isinstance(value, int) for value in state))


    def test_state_includes_macro_pellets_and_exact_ghost_offsets(self):
        game = Game(QLearningAgent(seed=31))
        state = game.observe()
        self.assertEqual(len(state), 18)
        self.assertEqual(state[0:2], game.pacman.position)
        # Chaser and ambusher offsets are represented separately.
        self.assertEqual(state[9:13], (7, 0, 12, 0))
        self.assertEqual(state[13:17], (0, 7, 12, 0))

    def test_action_prior_only_scores_safe_legal_actions(self):
        game = Game(QLearningAgent(seed=32))
        safe_actions = game.safe_action_indices()
        scores = game.action_prior_scores(safe_actions)
        self.assertEqual(set(scores), set(safe_actions))
        self.assertTrue(all(isinstance(value, float) for value in scores.values()))

    def test_ghost_transition_distribution_is_normalized(self):
        game = Game(QLearningAgent(seed=33))
        distribution = game._ghost_transition_distribution(
            game.chaser, game.chaser.position, game.chaser.speed_progress,
            game.pacman.position, game.pacman.direction,
        )
        self.assertAlmostEqual(sum(item[2] for item in distribution), 1.0)

    def test_recent_win_rate_uses_the_last_twenty_results(self):
        game = Game(QLearningAgent(seed=4))
        game.episode_history = [True, False, True, True]
        self.assertAlmostEqual(game.recent_win_rate, 75.0)


if __name__ == "__main__":
    unittest.main()
