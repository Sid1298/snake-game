"""Pygame view of a SnakeEnv, styled like game.py. Never touches the score database."""
import pygame

BLOCK_SIZE = 20

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (213, 50, 80)
BLACK = (0, 0, 0)


class Renderer:
    def __init__(self, env, fps=10, caption="Snake Game - AI agent"):
        pygame.init()
        self.env = env
        self.fps = fps
        self.screen = pygame.display.set_mode((env.cols * BLOCK_SIZE, env.rows * BLOCK_SIZE))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)

    def draw(self, status=""):
        """Draw one frame. Returns False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BLACK)
        for x, y in self.env.body:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        fx, fy = self.env.food
        pygame.draw.rect(self.screen, RED, pygame.Rect(fx * BLOCK_SIZE, fy * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = f"Score: {self.env.score}" + (f"   {status}" if status else "")
        self.screen.blit(self.font.render(text, True, WHITE), (10, 10))
        pygame.display.update()
        self.clock.tick(self.fps)
        return True

    def close(self):
        pygame.quit()
