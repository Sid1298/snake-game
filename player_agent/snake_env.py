"""Headless replica of the rules in game.py, used for training agents.

This module intentionally has no pygame or database imports, so nothing an
agent does here can ever reach the leaderboard.
"""
import random
from collections import deque

import numpy as np

# Mirrors game.py: a 450x300 window of 20px blocks
GRID_COLS = 450 // 20
GRID_ROWS = 300 // 20

# Clockwise order so that turning is just an index shift (screen y grows downward)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
UP, RIGHT, DOWN, LEFT = range(4)

# Actions are relative to the current heading, so the agent can never
# reverse into itself (game.py's key handler blocks that too)
STRAIGHT, TURN_RIGHT, TURN_LEFT = range(3)
ACTION_TURNS = (0, 1, -1)
N_ACTIONS = len(ACTION_TURNS)

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0


class SnakeEnv:
    def __init__(self, cols=GRID_COLS, rows=GRID_ROWS, max_idle_steps=None, seed=None):
        self.cols, self.rows = cols, rows
        self.n_cells = cols * rows
        # game.py has no time limit, but an agent can circle forever without eating
        self.max_idle_steps = max_idle_steps or 2 * self.n_cells
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        # Same start as game.py: head at pixel (100, 60) facing right, two blocks behind
        self.direction = RIGHT
        self.body = deque([(5, 3), (4, 3), (3, 3)])
        self.body_set = set(self.body)
        self.score = 0
        self.steps = 0
        self.idle_steps = 0
        self.done = False
        self._spawn_food()
        return self.observe()

    def _spawn_food(self):
        # Like game.py, food may land on the snake's body
        self.food = (self.rng.randrange(self.cols), self.rng.randrange(self.rows))

    def _move(self, cell, direction):
        dx, dy = DIRECTIONS[direction]
        return ((cell[0] + dx) % self.cols, (cell[1] + dy) % self.rows)

    def step(self, action):
        """Advance one tick. Returns (observation, reward, done, info)."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        self.direction = (self.direction + ACTION_TURNS[action]) % 4
        head = self._move(self.body[0], self.direction)
        self.steps += 1
        self.idle_steps += 1

        # game.py inserts the new head before the tail moves, so running into the
        # current tail cell is fatal there as well
        if head in self.body_set:
            self.done = True
            return self.observe(), REWARD_DEATH, True, self._info("self_hit")

        self.body.appendleft(head)
        self.body_set.add(head)

        if head == self.food:
            self.score += 1
            self.idle_steps = 0
            reward = REWARD_FOOD
            self._spawn_food()
        else:
            self.body_set.discard(self.body.pop())
            reward = 0.0

        if self.idle_steps >= self.max_idle_steps:
            self.done = True
            return self.observe(), REWARD_DEATH, True, self._info("starved")

        return self.observe(), reward, False, self._info(None)

    def _info(self, reason):
        return {"score": self.score, "steps": self.steps, "death": reason}

    # ----- observation ---------------------------------------------------

    def _wrapped_delta(self, a, b, size):
        # Shortest signed distance from a to b on a wrap-around axis
        d = (b - a) % size
        return d - size if d > size / 2 else d

    def _reachable_cells(self, start):
        """Flood fill from start, treating the body (minus the tail, which will move) as walls."""
        if start in self.body_set:
            return 0
        walls = set(self.body)
        walls.discard(self.body[-1])
        seen = {start}
        frontier = [start]
        while frontier:
            cell = frontier.pop()
            for d in range(4):
                nxt = self._move(cell, d)
                if nxt not in seen and nxt not in walls:
                    seen.add(nxt)
                    frontier.append(nxt)
        return len(seen)

    def _ray_to_body(self, direction):
        """Steps until the body is hit when looking straight along direction (normalized)."""
        limit = self.cols if DIRECTIONS[direction][1] == 0 else self.rows
        cell = self.body[0]
        for dist in range(1, limit):
            cell = self._move(cell, direction)
            if cell in self.body_set:
                return dist / limit
        return 1.0

    def observe(self):
        """Feature vector describing the board from the snake's point of view."""
        head = self.body[0]
        rel_dirs = [(self.direction + t) % 4 for t in ACTION_TURNS]
        free_cells = max(self.n_cells - len(self.body), 1)

        danger = []
        space = []
        rays = []
        for d in rel_dirs:
            nxt = self._move(head, d)
            danger.append(float(nxt in self.body_set))
            space.append(min(self._reachable_cells(nxt) / free_cells, 1.0))
            rays.append(self._ray_to_body(d))

        # Food position rotated into the snake's frame (forward / rightward components)
        dx = self._wrapped_delta(head[0], self.food[0], self.cols)
        dy = self._wrapped_delta(head[1], self.food[1], self.rows)
        fx, fy = DIRECTIONS[self.direction]
        rx, ry = DIRECTIONS[(self.direction + 1) % 4]
        forward = dx * fx + dy * fy
        rightward = dx * rx + dy * ry
        half = max(self.cols, self.rows) / 2

        heading = [0.0] * 4
        heading[self.direction] = 1.0

        features = (
            danger
            + space
            + rays
            + [
                float(forward > 0), float(forward < 0),
                float(rightward > 0), float(rightward < 0),
                forward / half, rightward / half,
            ]
            + heading
            + [len(self.body) / self.n_cells]
        )
        return np.asarray(features, dtype=np.float32)


OBSERVATION_SIZE = len(SnakeEnv().observe())
