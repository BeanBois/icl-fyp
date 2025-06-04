import math
import random
from collections import defaultdict
import sys
from enum import Enum 

import pygame
import numpy as np 



# DEBUGGING
import pdb

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 300
WHITE = (255, 255, 255) # WHITE SPACE
BLACK = (0, 0, 0) # EDGE FOR GOAL
RED = (255, 0, 0) # COLOR FOR OBSTACLES
GREEN = (0, 255, 0) # COLOR FOR EDIBLES
BLUE = (0, 0, 255) # COLOR FOR PLAYER NOT IN EATING STATE
PURPLE = (128, 0, 128) # COLOR FOR PLAYER IN EATING STATE
YELLOW = (255, 255, 0) # COLOR FOR GOAL

NUM_OF_EDIBLE_OBJECT = 1
NUM_OF_OBSTACLES = 1

class PlayerState(Enum):
    NOT_EATING = 1
    EATING = 2

class Player:
    
    def __init__(self, x, y, state : PlayerState = PlayerState.NOT_EATING):
        self.x = x
        self.y = y
        self.angle = 0  # Facing direction in degrees
        self.size = 15
        self.speed = 3
        self.rotation_speed = 5
        self.state = state
        
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
    
    def alternate_state(self):
        if self.state is PlayerState.EATING:
            self.state = PlayerState.NOT_EATING 
        elif self.state is PlayerState.NOT_EATING:
            self.state = PlayerState.EATING
        else:
            pass
 
    def get_rect(self):
        return pygame.Rect(self.x - self.size, self.y - self.size, 
                          self.size * 2, self.size * 2)
    
    # TL, TR, BR, BL, Centre dictonary
    def get_pos(self):
        return {
            'top-left': (self.x, self.y),
            'top-right' : (self.x + self.size, self.y),
            'bot-right' : (self.x + self.size, self.y + self.size),
            'bot-left' : (self.x, self.y + self.size),
            'centre' : (self.x + self.size // 2, self.y + self.size // 2),
            'orientation' : self.angle
        }

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

        if self.state is PlayerState.NOT_EATING:
            pygame.draw.polygon(screen, BLUE, points)
        else:
            pygame.draw.polygon(screen, PURPLE, points)

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

    # TL, TR, BR, BL, CENTROID dictonary
    def get_pos(self):
        return {
            'top-left': (self.x - self.width//2, self.y - self.height//2),
            'top-right': (self.x + self.width//2, self.y - self.height//2),
            'bot-right': (self.x + self.width//2, self.y + self.height//2),
            'bot-left': (self.x - self.width//2, self.y + self.height//2),
            'centre': (self.x, self.y),
        }

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
    
    # TL, TR, BR, BL, CENTROID dictonary
    def get_pos(self):
        return {
            'top-left': (self.x - self.width//2, self.y - self.height//2),
            'top-right': (self.x + self.width//2, self.y - self.height//2),
            'bot-right': (self.x + self.width//2, self.y + self.height//2),
            'bot-left': (self.x - self.width//2, self.y + self.height//2),
            'centre': (self.x, self.y),
        }

class GameObjective(Enum):
    EAT_ALL = 1
    REACH_GOAL = 2

    
class Game:
    
    def __init__(self, num_edibles = NUM_OF_EDIBLE_OBJECT, num_obstacles = NUM_OF_OBSTACLES):
        self.num_edibles = num_edibles
        self.num_obstacles = num_obstacles
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
        self.objective = random.choice([GameObjective.EAT_ALL, GameObjective.REACH_GOAL])
        
        # Create edible objects if objective is eat all edibles
        if self.objective is GameObjective.EAT_ALL:
            for _ in range(self.num_edibles):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, SCREEN_HEIGHT - 50)
                # Make sure edibles don't spawn too close to player
                while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 100:
                    x = random.randint(50, SCREEN_WIDTH - 50)
                    y = random.randint(50, SCREEN_HEIGHT - 50)
                
                width = random.randint(15, 25)
                height = random.randint(15, 25)
                self.edibles.append(EdibleObject(x, y, width, height))
        
        # Create obstacles and set goal position if objective is reach goal
        if self.objective is GameObjective.REACH_GOAL:
            for _ in range(self.num_obstacles):
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, SCREEN_HEIGHT - 50)
                # Make sure obstacles don't spawn too close to player
                while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 150:
                    x = random.randint(50, SCREEN_WIDTH - 50)
                    y = random.randint(50, SCREEN_HEIGHT - 50)
                
                width = random.randint(30, 50)
                height = random.randint(30, 50)
                self.obstacles.append(Obstacle(x, y, width, height))
        
            self.goal_position = (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and (self.game_over or self.game_won):
                    self.restart_game()
                elif event.key == pygame.K_SPACE and not self.game_over and not self.game_won:
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
        if self.objective == GameObjective.REACH_GOAL and self.goal_position:
            pygame.draw.rect(self.screen, YELLOW, 
                           (self.goal_position[0] - 25, self.goal_position[1] - 25, 50, 50))
            pygame.draw.rect(self.screen, BLACK, 
                           (self.goal_position[0] - 25, self.goal_position[1] - 25, 50, 50), 3)
        

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
        self.player = Player(100, 100)
        self.edibles = []
        self.obstacles = []
        self.goal_position = None
        self.game_over = False
        self.game_won = False
        self.setup_game()
    
    # RUNS THE WHOLE GAME 
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

    def get_screen_pixels(self):
        pixels = [
            [ np.array(self.screen.get_at((x,y))[:3]) for y in range(SCREEN_HEIGHT)] for x in range(SCREEN_WIDTH)
        ]

        return np.array(pixels)


# make GameInteraface s.t an action can be done with Translation + Rotation matrix
class GameInterface:

    def __init__(self,num_edibles = NUM_OF_EDIBLE_OBJECT, num_obstacles = NUM_OF_OBSTACLES, num_sampled_points = 10):
        self.game = Game(num_edibles=num_edibles, num_obstacles=num_obstacles)
        self.running = True
        self.t = 0
        self.num_sampled_points = num_sampled_points

    def start_game(self):
        self.game.draw()
        self.game.clock.tick(60)
        obs = self.get_obs()
        return obs 
  
    def restart_game(self):
        self.game.restart_game()
        self.running = True
        self.t = 0

    # we do furthest point sampling algorithm to retrieve a cloud of point clouds
    # These sampled ppoint clouds is then used to create a graph representation 
    def get_obs(self):
        agent_pos = self._get_agent_pos()
        # Since our 'point clouds' are represented as pixels in a 2d grid, our dense point cloud will be a 2d matrix of Screen-width x Screen-height
        raw_dense_point_clouds = self.game.get_screen_pixels()
        raw_coords = np.array([[(x,y) for y in range(SCREEN_HEIGHT) ]  for x in range(SCREEN_WIDTH)])

        # To ensure that only objects point clouds are picked up, we remove all white pixels and agent pixels
        mask = ~np.all((raw_dense_point_clouds == [255, 255, 255]) | 
                    (raw_dense_point_clouds == [128, 0, 128]) |
                    (raw_dense_point_clouds == [0, 0, 255])
                    , axis=2)
        valid_points = np.where(mask)
        dense_point_clouds = raw_dense_point_clouds[valid_points]
        coords = raw_coords[valid_points]

        # To simulate FPSA, we randomly select an initial point from coords ranging from p in (SCREEN_WIDTH X SCREEN_HEIGHT)  
        selected_indices = []
        initial_coord = random.randint(0, len(coords) - 1)
        selected_indices.append(initial_coord)

        # then perform FPSA to collect M points
        for _ in range(self.num_sampled_points-1):
            if len(selected_indices) >= len(coords):
                break

            selected_coords = coords[selected_indices]
        
            # Calculate minimum distance from each unselected point to nearest selected point
            max_min_distance = -1
            best_idx = -1
            
            for i, coord in enumerate(coords):
                if i in selected_indices:
                    continue
                    
                # Calculate distances to all selected points
                distances = np.linalg.norm(selected_coords - coord, axis=1)
                min_distance = np.min(distances)
                
                # Keep track of point with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)

        # Categorise/Segment M points according to colour (sidestepping geometric encoder)
        selected_coords = coords[selected_indices]
        selected_colors = dense_point_clouds[selected_indices]
        color_segments = defaultdict(list)
        agent_state = None
        if BLUE in selected_colors:
            agent_state = 'not-eating'
        elif PURPLE in selected_colors:
            agent_state = 'eating'
        
        for i, (coord, color) in enumerate(zip(selected_coords, selected_colors)):
            # Convert RGB to a hashable tuple for grouping
            # color_key = tuple(color.astype(int))
            color_key = None
            color = tuple(color)
            if color == BLACK or color == YELLOW:
                color_key = 'goal'
            elif color == GREEN:
                color_key = 'edible'
            elif color == RED:
                color_key = 'obstacle'
            else:
                color_key = 'unknown'
            color_segments[color_key].append({
                'coord': coord,
                'index': i,
                'color': color
            })        

        return {
            'point-clouds': dict(color_segments),
            'agent-pos' : agent_pos,
            'agent-state' : agent_state,
            'done': self.running,
            'time' : self.t
        }

    def step(self):
        if self.running:
            self.t += 1
            self.running = self.game.handle_events()
            self.game.update()
            self.game.draw()
            self.game.clock.tick(60)
            obs = self.get_obs()
            return obs 
        else:
            pygame.quit()
            sys.exit()

    # return dictionary of point clouds taken from screen
    # for now we just return the corners and centroid of each object
    # each point cloud encodes the position and colour of the particular pixel 
    # belows will be implemented in another function by class Subgraph and Graph within the agent file 
        # the collected point clouds are used to construct a sub-graph which represents the object geometry/shape 
            # this edge type will be intra-object edge-type, and its feature also includes a distance based on euc-distance for now
        # then to construct the local graph, corners of each object are fully connected to all other corners of other object, and centroid are connected to other centroids
            # this edge type will be inter-object edge-type, and its feature also includes a distance based on euc-distance for now
    def _get_game_obj_pos(self):
        # first get player 
        player_pos = self.game.player.get_pos()

        # then get obstacles and object 
        obstacles = [obj.get_pos() for obj in self.game.obstacles]
        edibles = [edi.get_pos() for edi in self.game.edibles]
        objs_pos = obstacles + edibles

        return {
            'player' : player_pos,
            'objects' : objs_pos,
            'done' : self.running,
            'objective' : self.game.objective
        }
    
    def _get_agent_pos(self):
        player_pos = self.game.player.get_pos()
        tl = np.array(player_pos['top-left'])
        tr = np.array(player_pos['top-right'])
        br = np.array(player_pos['bot-right'])
        bl = np.array(player_pos['bot-left'])
        center = np.array(player_pos['centre'])
        ori = player_pos['orientation']

        # player is a triangle so i want to capture the 3 edges of the triangle
        # at player_ori == 0 degree, edges == (tl, bl, (tr+br)//2)
        tri_points = np.array([tl, bl, (tr+br)//2])
        R = self._rotation_matrix_2d(ori) 

        # rotate around center
        translated = tri_points - center
        rotated = (R @ translated.T).T
        final = rotated + center
        player_pos = np.vstack([final,center])


        return player_pos
    
    def _rotation_matrix_2d(self, theta):
        theta = theta/180 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s],
                        [s,  c]])


# Run the game
if __name__ == "__main__":
    game_interface = GameInterface()
    game_interface.start_game()
    game_interface.step()
    