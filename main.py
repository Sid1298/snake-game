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

# Function to display text at the top left corner (for score)
def display_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

# Function to display text centered on the screen
def display_centered_text(text, font, color, y_offset=0):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    
    # Center the text on the screen
    text_rect.center = (WIDTH // 2, HEIGHT // 2 + y_offset)
    
    screen.blit(text_surface, text_rect)
