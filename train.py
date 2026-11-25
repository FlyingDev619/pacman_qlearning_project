"""Train Pac-Man and stop when a frozen-policy validation reaches the target.

Usage:
    python train.py --episodes 5000 --eval-episodes 100 --target 90
"""
import argparse
import os
import random

from agent import QLearningAgent
from game import Game


def evaluate(agent, seeds, episodes):
    previous_epsilon, previous_alpha = agent.epsilon, agent.alpha
    previous_read_only = agent.read_only
    agent.epsilon, agent.alpha, agent.read_only = 0.0, 0.0, True
    wins = total = 0
    try:
        for seed in seeds:
            random.seed(seed)
            game = Game(agent)
            for _ in range(episodes):
                while not game.terminal:
                    game.step()
                wins += int(game.episode_history[-1])
                total += 1
                game.update(2.0)
    finally:
        agent.epsilon, agent.alpha = previous_epsilon, previous_alpha
        agent.read_only = previous_read_only
    return 100.0 * wins / max(1, total), wins, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000,
                        help="maximum episodes unless --until-target is enabled")
    parser.add_argument("--until-target", action="store_true",
                        help="keep training until frozen-policy validation is strictly above --target")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--target", type=float, default=90.0)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--output", default="q_table.json")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--resume", action="store_true", help="continue from --output if it is valid")
    parser.add_argument(
        "--resume-epsilon", type=float, default=0.25,
        help="minimum exploration rate when resuming a nearly converged checkpoint",
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+",
                        default=[90001, 90002, 90003, 90004, 90005])
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes, args.check_every) <= 0:
        parser.error("episode counts and check interval must be positive")

    random.seed(args.seed)
    agent = QLearningAgent(seed=args.seed)
    resumed = False
    if args.resume and os.path.exists(args.output):
        try:
            agent.load(args.output)
            if not 0.0 <= args.resume_epsilon <= 1.0:
                parser.error("--resume-epsilon must be between 0 and 1")
            old_epsilon = agent.epsilon
            agent.epsilon = max(agent.epsilon, args.resume_epsilon)
            resumed = True
            print(
                f"Resumed {len(agent.q_table)} states; exploration "
                f"{old_epsilon:.4f} -> {agent.epsilon:.4f}."
            )
        except (OSError, ValueError, KeyError):
            print("Existing Q-table is invalid or incompatible; starting fresh.")
    best_rate = -1.0
    if resumed:
        best_rate, baseline_wins, baseline_total = evaluate(
            agent, args.eval_seeds, args.eval_episodes
        )
        print(
            f"checkpoint baseline={baseline_wins}/{baseline_total} "
            f"({best_rate:.2f}%); only better checkpoints will replace it.",
            flush=True,
        )
        if best_rate > args.target:
            print(
                f"Target already exceeded: {best_rate:.2f}% > {args.target:.2f}%."
            )
            return
        random.seed(args.seed)
    game = Game(agent)
    episode = 0
    while args.until_target or episode < args.episodes:
        episode += 1
        while not game.terminal:
            game.step()
        game.update(2.0)
        if episode % args.check_every == 0 or (
            not args.until_target and episode == args.episodes
        ):
            rate, wins, total = evaluate(agent, args.eval_seeds, args.eval_episodes)
            if rate > best_rate:
                agent.save(args.output)
                best_rate = rate
            print(
                f"training={episode}/{'until-target' if args.until_target else args.episodes} "
                f"validation={wins}/{total} ({rate:.2f}%) "
                f"states={len(agent.q_table)} epsilon={agent.epsilon:.4f}",
                flush=True,
            )
            if rate > args.target:
                print(f"Target reached: frozen-policy win rate {rate:.2f}% > {args.target:.2f}%.")
                return
    if not args.until_target:
        print(
            f"Target not reached; best validation rate was {best_rate:.2f}%. "
            f"Table: {args.output}"
        )


if __name__ == "__main__":
    main()
