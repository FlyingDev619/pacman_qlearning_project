"""Evaluate a saved Q-table without exploration or Q-value updates.

Example:
    python evaluate.py --episodes 100 --seeds 78001 78002
"""

import argparse
import random
import statistics

from agent import QLearningAgent
from game import Game


def evaluate_seed(agent, seed, episodes):
    random.seed(seed)
    game = Game(agent)
    wins = 0
    pellets_collected = []
    steps = []

    for _ in range(episodes):
        while not game.terminal:
            game.step()
        wins += int(game.episode_history[-1])
        pellets_collected.append(len(game.maze.floor) - len(game.maze.pellets))
        steps.append(game.step_count)
        game.update(2.0)

    return {
        "seed": seed,
        "wins": wins,
        "episodes": episodes,
        "win_rate": 100.0 * wins / episodes,
        "avg_pellets": statistics.mean(pellets_collected),
        "avg_steps": statistics.mean(steps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtable", default="q_table.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[78001, 78002])
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    agent = QLearningAgent(seed=0)
    agent.load(args.qtable)
    agent.epsilon = 0.0
    agent.alpha = 0.0
    agent.read_only = True

    total_wins = 0
    total_episodes = 0
    for seed in args.seeds:
        result = evaluate_seed(agent, seed, args.episodes)
        total_wins += result["wins"]
        total_episodes += result["episodes"]
        print(
            f"seed={result['seed']}  "
            f"wins={result['wins']}/{result['episodes']}  "
            f"win_rate={result['win_rate']:.1f}%  "
            f"avg_pellets={result['avg_pellets']:.1f}  "
            f"avg_steps={result['avg_steps']:.1f}"
        )

    rate = 100.0 * total_wins / total_episodes
    print(
        f"combined={total_wins}/{total_episodes}  "
        f"win_rate={rate:.1f}%"
    )


if __name__ == "__main__":
    main()
