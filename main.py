import sys
import pygame
import random

# Initialize
pygame.init()

# Constants
WIDTH, HEIGHT = 900, 600
BLOCK_SIZE = 20
FPS = 10

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (213, 50, 80)
BLACK = (0, 0, 0)
GREY = (30, 30, 30)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake Game')

# Intiating clock
clock = pygame.time.Clock()

# Font for displaying messages
small_font = pygame.font.SysFont('arial', 15)
font = pygame.font.SysFont('arial', 25)
big_font = pygame.font.SysFont('arial', 40)

# Snake setup
snake_pos = [100, 60]
snake_body = [[100, 60], [80, 60], [60, 60]]
direction = 'RIGHT'
change_to = direction

# Food setup
food_pos = [random.randrange(0, WIDTH, BLOCK_SIZE), random.randrange(0, HEIGHT, BLOCK_SIZE)]
food_spawn = True

# Score setup
score = 0
