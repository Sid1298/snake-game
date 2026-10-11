"""Train a DQN agent on the snake game from scratch.

    python -m player_agent.train --episodes 1000
    python -m player_agent.train --episodes 200 --render     # watch it learn (slow)

Training runs on the headless SnakeEnv, so bot scores are never written to the
leaderboard database.
"""
import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch

from player_agent.agent import DQNAgent
from player_agent.snake_env import N_ACTIONS, OBSERVATION_SIZE, SnakeEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--name", default="dqn", help="checkpoint name inside player_agent/models/")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="show the game window while training")
    parser.add_argument("--fps", type=int, default=60, help="render speed when --render is set")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    latest_path = os.path.join(MODEL_DIR, f"{args.name}_latest.pt")
    best_path = os.path.join(MODEL_DIR, f"{args.name}_best.pt")
    log_path = os.path.join(MODEL_DIR, f"{args.name}_log.csv")

    env = SnakeEnv(seed=args.seed)
    agent = DQNAgent(OBSERVATION_SIZE, N_ACTIONS)

    start_episode, best_avg = 0, float("-inf")
    if args.resume and os.path.exists(latest_path):
        checkpoint = agent.load(latest_path)
        start_episode = checkpoint.get("episode", 0)
        best_avg = checkpoint.get("best_avg", best_avg)
        print(f"Resumed from {latest_path} at episode {start_episode} (step {agent.total_steps})")

    renderer = None
    if args.render:
        from player_agent.render import Renderer

        renderer = Renderer(env, fps=args.fps, caption="Snake Game - AI training")

    recent_scores = deque(maxlen=100)
    best_score = 0
    log_mode = "a" if args.resume and os.path.exists(log_path) else "w"
    started = time.time()

    with open(log_path, log_mode, newline="") as log_file:
        log = csv.writer(log_file)
        if log_mode == "w":
            log.writerow(["episode", "score", "steps", "death", "epsilon", "avg100"])

        for episode in range(start_episode + 1, start_episode + args.episodes + 1):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state = next_state

                if renderer and not renderer.draw(f"Episode {episode}  eps {agent.epsilon:.2f}"):
                    renderer.close()
                    renderer = None

            recent_scores.append(info["score"])
            best_score = max(best_score, info["score"])
            avg = sum(recent_scores) / len(recent_scores)
            log.writerow([episode, info["score"], info["steps"], info["death"], round(agent.epsilon, 4), round(avg, 3)])

            # Only judge "best" once the agent is mostly exploiting what it learned
            if len(recent_scores) == recent_scores.maxlen and agent.epsilon < 0.1 and avg > best_avg:
                best_avg = avg
                agent.save(best_path, episode=episode, best_avg=best_avg)

            if episode % args.log_every == 0:
                log_file.flush()
                agent.save(latest_path, episode=episode, best_avg=best_avg)
                print(
                    f"ep {episode:5d} | score {info['score']:3d} | avg100 {avg:6.2f} | best {best_score:3d} "
                    f"| eps {agent.epsilon:.3f} | steps {agent.total_steps} | {time.time() - started:6.0f}s"
                )

    agent.save(latest_path, episode=episode, best_avg=best_avg)
    print(f"Done. Best 100-episode average: {best_avg:.2f}. Checkpoints in {MODEL_DIR}")
    if renderer:
        renderer.close()


if __name__ == "__main__":
    main()
