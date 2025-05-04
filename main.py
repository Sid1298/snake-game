import sys
import pygame
import random
from datetime import datetime
from database.database_setup import add_new_score, get_top_5_scores, User


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
# Main game loop
def game_loop():
    player_name = capture_username()
    global snake_pos, snake_body, direction, change_to, food_pos, food_spawn, score

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

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Control snake with arrow keys
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != 'DOWN':
                    change_to = 'UP'
                elif event.key == pygame.K_DOWN and direction != 'UP':
                    change_to = 'DOWN'
                elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                    change_to = 'LEFT'
                elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                    change_to = 'RIGHT'

        # Update direction
        direction = change_to

        # Move snake head
        if direction == 'UP':
            snake_pos[1] -= BLOCK_SIZE
        elif direction == 'DOWN':
            snake_pos[1] += BLOCK_SIZE
        elif direction == 'LEFT':
            snake_pos[0] -= BLOCK_SIZE
        elif direction == 'RIGHT':
            snake_pos[0] += BLOCK_SIZE

        # Wrap snake around the screen (spherical world)
        if snake_pos[0] >= WIDTH:  # Goes off right side
            snake_pos[0] = 0
        elif snake_pos[0] < 0:  # Goes off left side
            snake_pos[0] = WIDTH - BLOCK_SIZE

        if snake_pos[1] >= HEIGHT:  # Goes off bottom
            snake_pos[1] = 0
        elif snake_pos[1] < 0:  # Goes off top
            snake_pos[1] = HEIGHT - BLOCK_SIZE

        # Insert new head
        snake_body.insert(0, list(snake_pos))

        # Self-hit condition
        if snake_pos in snake_body[1:]:
            running = False  # End the game if the snake hits itself

        # Check if food is eaten
        if snake_pos == food_pos:
            food_spawn = False
            score += 1  # Increase score
        else:
            snake_body.pop()

        # Spawn new food if needed
        if not food_spawn:
            food_pos = [random.randrange(0, WIDTH, BLOCK_SIZE), random.randrange(0, HEIGHT, BLOCK_SIZE)]
            food_spawn = True

        # Drawing
        screen.fill(BLACK)

        # Draw snake
        for block in snake_body:
            pygame.draw.rect(screen, GREEN, pygame.Rect(block[0], block[1], BLOCK_SIZE, BLOCK_SIZE))

        # Draw food
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], BLOCK_SIZE, BLOCK_SIZE))

        # Display score at top left corner
        display_text(f"Score: {score}", font, WHITE, 10, 10)

        # Update the display
        pygame.display.update()

        # Control game speed
        clock.tick(FPS)

    # Game Over screen (centered text)
    display_centered_text(f"Game Over! Your score: {score}", big_font, RED, y_offset=-70)
    display_centered_text("Press Enter to Play Again or ESC to Quit", font, WHITE, y_offset=0)
    
    pygame.display.update()

    player = User(name=player_name, score=score, playdate=datetime.now())

    add_new_score(player)
    top5=[f"{player.name} - {player.score}" for player in get_top_5_scores()]
    display_centered_text("Top Scores:", font, WHITE, y_offset=30)
    pygame.display.update()
    for index, value in enumerate(top5):
        display_centered_text(value, font=small_font, color=WHITE, y_offset=75+15*index)
        pygame.display.update()

    # Wait for restart or quit
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Restart game
                    game_loop()  # Restart the game
                elif event.key == pygame.K_ESCAPE:  # Quit game
                    pygame.quit()
                    quit()

# Start the game loop
game_loop()
