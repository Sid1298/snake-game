# Player agent

A Deep Q-Network (DQN) agent that learns snake from scratch: it starts with random weights and only ever sees the reward signal (+10 for food, -10 for dying).

Agents play on `snake_env.py`, a headless replica of `game.py`'s rules (wrap-around grid, same start position, hitting your own body ends the game). Nothing in this folder imports `database`, so bot games are never written to the leaderboard.

## Setup

```bash
venv/bin/pip install -r player_agent/requirements.txt
```

## Train

Run from the repo root:

```bash
venv/bin/python -m player_agent.train --episodes 1000          # headless, fast
venv/bin/python -m player_agent.train --episodes 200 --render  # watch it learn
venv/bin/python -m player_agent.train --episodes 500 --resume  # keep training
```

Checkpoints and a per-episode CSV log go to `player_agent/models/` (gitignored):
- `dqn_latest.pt` is the most recent weights, used by `--resume`.
- `dqn_best.pt` is the weights with the best 100-game average score.

## Watch / benchmark

```bash
venv/bin/python -m player_agent.play                         # watch 5 games
venv/bin/python -m player_agent.play --headless --games 100  # score stats only
```

## What the agent sees

The agent sees 20 numbers, all relative to the snake's heading (straight / right / left):
- whether the next cell in each direction is body
- how much of the board is reachable after each move (flood fill), which helps it avoid boxing itself in
- distance to the nearest body segment along each direction
- where the food is, using the shortest wrap-around distance
- the current heading and the snake's length

It has three actions: go straight, turn right, or turn left.
