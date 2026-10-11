"""Watch (or benchmark) a trained agent. Scores are never saved to the leaderboard.

    python -m player_agent.play                      # watch the best checkpoint
    python -m player_agent.play --headless --games 100
"""
import argparse
import os

from player_agent.agent import DQNAgent
from player_agent.snake_env import N_ACTIONS, OBSERVATION_SIZE, SnakeEnv
from player_agent.train import MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.join(MODEL_DIR, "dqn_best.pt"))
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--headless", action="store_true", help="no window, just print scores")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    agent = DQNAgent(OBSERVATION_SIZE, N_ACTIONS)
    agent.load(args.model)
    agent.policy.eval()

    env = SnakeEnv(seed=args.seed)
    renderer = None
    if not args.headless:
        from player_agent.render import Renderer

        renderer = Renderer(env, fps=args.fps)

    scores = []
    for game in range(1, args.games + 1):
        state = env.reset()
        done = False
        while not done:
            state, _, done, info = env.step(agent.act(state, explore=False))
            if renderer and not renderer.draw(f"Game {game}/{args.games}"):
                renderer.close()
                return
        scores.append(info["score"])
        print(f"game {game:4d} | score {info['score']:3d} | steps {info['steps']:5d} | {info['death']}")

    print(f"\n{len(scores)} games | mean {sum(scores) / len(scores):.2f} | max {max(scores)} | min {min(scores)}")
    if renderer:
        renderer.close()


if __name__ == "__main__":
    main()
