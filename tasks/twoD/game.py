import pygame
import math
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # Facing direction in degrees
        self.size = 15
        self.speed = 3
        self.rotation_speed = 5
        
    def move_forward(self):
        # Convert angle to radians and move in facing direction
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        
        # Keep player within screen bounds
        self.x = max(self.size, min(SCREEN_WIDTH - self.size, self.x))
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))
    
    def move_backward(self):
        rad = math.radians(self.angle)
        self.x -= self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        
        # Keep player within screen bounds
        self.x = max(self.size, min(SCREEN_WIDTH - self.size, self.x))
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))
    
    def rotate_left(self):
        self.angle -= self.rotation_speed
        self.angle %= 360
    
    def rotate_right(self):
        self.angle += self.rotation_speed
        self.angle %= 360
    
    def get_rect(self):
        return pygame.Rect(self.x - self.size, self.y - self.size, 
                          self.size * 2, self.size * 2)
    
    def draw(self, screen):
        # Draw player as a triangle pointing in the direction it's facing
        rad = math.radians(self.angle)
        
        # Calculate triangle points
        front_x = self.x + self.size * math.cos(rad)
        front_y = self.y + self.size * math.sin(rad)
        
        back_left_x = self.x + self.size * math.cos(rad + 2.4)
        back_left_y = self.y + self.size * math.sin(rad + 2.4)
        
        back_right_x = self.x + self.size * math.cos(rad - 2.4)
        back_right_y = self.y + self.size * math.sin(rad - 2.4)
        
        points = [(front_x, front_y), (back_left_x, back_left_y), (back_right_x, back_right_y)]
        pygame.draw.polygon(screen, BLUE, points)

class EdibleObject:
    def __init__(self, x, y, width=20, height=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.eaten = False
    
    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                          self.width, self.height)
    
    def draw(self, screen):
        if not self.eaten:
            pygame.draw.ellipse(screen, GREEN, self.get_rect())

class Obstacle:
    def __init__(self, x, y, width=40, height=40):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                          self.width, self.height)
    
    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.get_rect())

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Game - Eat or Avoid")
        self.clock = pygame.time.Clock()
        
        # Initialize game objects
        self.player = Player(100, 100)
        self.edibles = []
        self.obstacles = []
        self.goal_position = None
        
        # Game state
        self.objective = None  # "eat_all" or "reach_goal"
        self.game_over = False
        self.game_won = False
        self.font = pygame.font.Font(None, 36)
        
        self.setup_game()
    
    def setup_game(self):
        # Randomly choose objective
        self.objective = random.choice(["eat_all", "reach_goal"])
        
        # Create edible objects
        for _ in range(5):
            x = random.randint(50, SCREEN_WIDTH - 50)
            y = random.randint(50, SCREEN_HEIGHT - 50)
            # Make sure edibles don't spawn too close to player
            while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 100:
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, SCREEN_HEIGHT - 50)
            
            width = random.randint(15, 25)
            height = random.randint(15, 25)
            self.edibles.append(EdibleObject(x, y, width, height))
        
        # Create obstacles
        for _ in range(4):
            x = random.randint(50, SCREEN_WIDTH - 50)
            y = random.randint(50, SCREEN_HEIGHT - 50)
            # Make sure obstacles don't spawn too close to player
            while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 150:
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, SCREEN_HEIGHT - 50)
            
            width = random.randint(30, 50)
            height = random.randint(30, 50)
            self.obstacles.append(Obstacle(x, y, width, height))
        
        # Set goal position if objective is "reach_goal"
        if self.objective == "reach_goal":
            self.goal_position = (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and (self.game_over or self.game_won):
                    self.restart_game()
        
        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        if not self.game_over and not self.game_won:
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.player.move_forward()
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.player.move_backward()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.player.rotate_left()
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.player.rotate_right()
        
        return True
    
    def update(self):
        if self.game_over or self.game_won:
            return
        
        # Check collision with obstacles
        player_rect = self.player.get_rect()
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle.get_rect()):
                self.game_over = True
                return
        
        # Check collision with edibles
        for edible in self.edibles:
            if not edible.eaten and player_rect.colliderect(edible.get_rect()):
                edible.eaten = True
        
        # Check win conditions
        if self.objective == "eat_all":
            if all(edible.eaten for edible in self.edibles):
                self.game_won = True
        elif self.objective == "reach_goal":
            if self.goal_position:
                goal_rect = pygame.Rect(self.goal_position[0] - 25, self.goal_position[1] - 25, 50, 50)
                if player_rect.colliderect(goal_rect):
                    self.game_won = True
    
    def draw(self):
        self.screen.fill(WHITE)
        
        # Draw all game objects
        for edible in self.edibles:
            edible.draw(self.screen)
        
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        self.player.draw(self.screen)
        
        # Draw goal if objective is "reach_goal"
        if self.objective == "reach_goal" and self.goal_position:
            pygame.draw.rect(self.screen, YELLOW, 
                           (self.goal_position[0] - 25, self.goal_position[1] - 25, 50, 50))
            pygame.draw.rect(self.screen, BLACK, 
                           (self.goal_position[0] - 25, self.goal_position[1] - 25, 50, 50), 3)
        
        # Draw UI
        objective_text = "Objective: "
        if self.objective == "eat_all":
            objective_text += "Eat all green objects!"
            eaten_count = sum(1 for edible in self.edibles if edible.eaten)
            progress_text = f"Eaten: {eaten_count}/{len(self.edibles)}"
        else:
            objective_text += "Avoid red obstacles and reach yellow goal!"
            progress_text = "Navigate carefully to the goal!"
        
        text_surface = pygame.font.Font(None, 24).render(objective_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))
        
        progress_surface = pygame.font.Font(None, 24).render(progress_text, True, BLACK)
        self.screen.blit(progress_surface, (10, 35))
        
        # Draw controls
        controls = "Controls: Arrow Keys/WASD to move and rotate"
        controls_surface = pygame.font.Font(None, 20).render(controls, True, BLACK)
        self.screen.blit(controls_surface, (10, SCREEN_HEIGHT - 25))
        
        # Draw game over/win message
        if self.game_over:
            message = "GAME OVER! You hit an obstacle. Press R to restart."
            text_surface = self.font.render(message, True, RED)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            pygame.draw.rect(self.screen, WHITE, text_rect.inflate(20, 10))
            pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 10), 2)
            self.screen.blit(text_surface, text_rect)
        
        elif self.game_won:
            message = "YOU WIN! Objective completed! Press R to restart."
            text_surface = self.font.render(message, True, GREEN)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            pygame.draw.rect(self.screen, WHITE, text_rect.inflate(20, 10))
            pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 10), 2)
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
    
    def restart_game(self):
        self.player = Player(100, 100)
        self.edibles = []
        self.obstacles = []
        self.goal_position = None
        self.game_over = False
        self.game_won = False
        self.setup_game()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

# Run the game
if __name__ == "__main__":
    game = Game()
    game.run()