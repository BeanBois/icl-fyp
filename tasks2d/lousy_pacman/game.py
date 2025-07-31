# should store Game Code only
from .game_aux import * 
import random

class Game:
    
    def __init__(self, num_edibles = NUM_OF_EDIBLE_OBJECT, num_obstacles = NUM_OF_OBSTACLES, num_goals = NUM_OF_GOALS, screen_width = SCREEN_WIDTH, screen_height = SCREEN_HEIGHT, objective = None, player_start_pos = (PLAYER_START_X, PLAYER_START_Y)):
        self.num_edibles = num_edibles
        self.num_obstacles = num_obstacles
        self.num_goals = num_goals
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("2D Game - Eat or Avoid")
        self.clock = pygame.time.Clock()
        
        # Initialize game objects
        self.player_start_pos = player_start_pos
        self.player = Player(self.player_start_pos[0], self.player_start_pos[1])
        self.edibles = []
        self.obstacles = []
        self.goal = None
        
        # Game state
        self.objective = objective  # "eat_all" or "reach_goal"
        self.game_over = False
        self.game_won = False
        # self.font = pygame.font.Font(None, 36)

        self.setup_game()

    def setup_game(self):
        # Randomly choose objective
        if self.objective is None:
            self.objective = random.choice([GameObjective.EAT_ALL, GameObjective.REACH_GOAL])
        
        # Create edible objects if objective is eat all edibles
        if self.objective is GameObjective.EAT_ALL:
            for _ in range(self.num_edibles):
                x = random.randint(50, self.screen_width - 50)
                y = random.randint(50, self.screen_height - 50)
                # Make sure edibles don't spawn too close to player
                while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 100:
                    x = random.randint(50, self.screen_width - 50)
                    y = random.randint(50, self.screen_height - 50)
                
                width = random.randint(15, 25)
                height = random.randint(15, 25)
                self.edibles.append(EdibleObject(x, y, width, height))
        
        # Create obstacles and set goal position if objective is reach goal
        if self.objective is GameObjective.REACH_GOAL:
            for _ in range(self.num_obstacles):
                x = random.randint(50, self.screen_width - 50)
                y = random.randint(50, self.screen_height - 50)
                # Make sure obstacles don't spawn too close to player
                while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 150:
                    x = random.randint(50,  self.screen_width - 50)
                    y = random.randint(50, self.screen_height - 50)
                
                width = random.randint(30, 50)
                height = random.randint(30, 50)
                self.obstacles.append(Obstacle(x, y, width, height))
        
            goal_position = (self.screen_width - 100 + np.random.randint(-25,25), self.screen_height - 100 + np.random.randint(-25,25))
            self.goal = Goal(goal_position[0], goal_position[1])

        # Command line print to inform player
        print("-" * 25)
        objective_str = None
        if self.objective is GameObjective.EAT_ALL:
            objective_str = "eat all food"
        else:
            objective_str = "reach goal"
        print(f"Welcome to the game, your objective is to {objective_str}")
        print("-" * 25)

    def handle_events(self):
        if self.game_over or self.game_won:
            return False
        for event in pygame.event.get():
            # if event.type == pygame.QUIT:
            #     return False
            if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_r and (self.game_over or self.game_won):
            #         self.restart_game()
                if event.key == pygame.K_SPACE and not self.game_over and not self.game_won:
                    self.player.alternate_state()
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

    def handle_action(self,action : Action):
        if self.game_over or self.game_won:
            return False
        
        # Handle continuous key presses
        self.player.move_with_action(action)
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
            if not edible.eaten and player_rect.colliderect(edible.get_rect()) and self.player.state is PlayerState.EATING:
                edible.eaten = True
        
        # Check win conditions
        if self.objective == GameObjective.EAT_ALL:
            if all(edible.eaten for edible in self.edibles):
                self.game_won = True
        elif self.objective == GameObjective.REACH_GOAL:
            if self.goal:
                goal_rect = self.goal.get_rect()
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
        if self.objective == GameObjective.REACH_GOAL and self.goal:
            self.goal.draw(self.screen)
        

        message = None 
        # Draw game over/win message
        if self.game_over:
            message = "GAME OVER! You hit an obstacle. Press R to restart."
            print(message)
        
        elif self.game_won:
            message = "YOU WIN! Objective completed! Press R to restart."
            print(message)
        
        pygame.display.flip()
    
    def restart_game(self):
        self.player = Player(self.player_start_pos[0], self.player_start_pos[1])

        self.edibles = []
        self.obstacles = []
        self.goal = None
        self.game_over = False
        self.game_won = False
        self.setup_game()
    
    def end_game(self):
        pygame.quit()

    # RUNS THE WHOLE GAME 
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


    def get_screen_pixels(self):
        pixels = [
            [ np.array(self.screen.get_at((x,y))[:3]) for y in range(self.screen_height)] for x in range(self.screen_width)
        ]
        return np.array(pixels)

