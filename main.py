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

# Pregame screen
input_box = pygame.Rect(150, 150, 300, 50)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color = color_inactive
active = False
user_text = ""

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

# Take user name input
def capture_username():
    global user_text, color, active, big_font
    while True:
        screen.fill(GREY)
        display_text("Enter your name", big_font, WHITE, 10, 10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Toggle active state if user clicks on the input box
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive

            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        # Break here to start the game
                        return user_text
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        user_text += event.unicode

        # Render the input text
        txt_surface = font.render(user_text, True, color)
        width = max(300, txt_surface.get_width() + 10)
        input_box.w = width
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 10))
        pygame.draw.rect(screen, color, input_box, 2)

        pygame.display.flip()
        clock.tick(30)
