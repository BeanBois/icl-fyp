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
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255) # WHITE SPACE
BLACK = (0, 0, 0) # EDGE FOR GOAL
RED = (255, 0, 0) # COLOR FOR OBSTACLES
GREEN = (0, 255, 0) # COLOR FOR EDIBLES
BLUE = (0, 0, 255) # COLOR FOR PLAYER NOT IN EATING STATE
PURPLE = (128, 0, 128) # COLOR FOR PLAYER IN EATING STATE
YELLOW = (255, 255, 0) # COLOR FOR GOAL

NUM_OF_EDIBLE_OBJECT = 1
NUM_OF_OBSTACLES = 1
NUM_OF_GOALS = 1

class PlayerState(Enum):
    NOT_EATING = 1
    EATING = 2

# we will refactor this when everything is done
class Action:
    def __init__(self, forward_movement, orientation, state_change):
        self.forward_movement = forward_movement
        self.orientation = orientation
        self.state_change = state_change
    
    # returns movement as a matrix
    def movement_as_matrix(self):
        # If you want the movement to be relative to the player's current orientation,
        # use the rotation to determine direction
        orientation = np.deg2rad(self.orientation)
        
        # Create movement vector in the direction of rotation
        # Since rotation 0 should point right (positive x), we use cos/sin directly
        movement_vector = self.forward_movement * np.array([np.cos(orientation), np.sin(orientation)])
        
        return movement_vector
    

class Player:
    
    def __init__(self, x, y, state : PlayerState = PlayerState.NOT_EATING, screen_width = SCREEN_WIDTH, screen_height = SCREEN_HEIGHT):
        self.x = x
        self.y = y
        self.angle = 0  # Facing direction in degrees
        self.size = 15
        self.speed = 3
        self.rotation_speed = 5
        self.state = state
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def move_forward(self):
        # Convert angle to radians and move in facing direction
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        
        # Keep player within screen bounds
        self.x = max(self.size, min(self.screen_width - self.size, self.x))
        self.y = max(self.size, min(self.screen_height - self.size, self.y))
    
    def move_backward(self):
        rad = math.radians(self.angle)
        self.x -= self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        
        # Keep player within screen bounds
        self.x = max(self.size, min(self.screen_width - self.size, self.x))
        self.y = max(self.size, min(self.screen_height - self.size, self.y))
    
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
    

    def move_with_action(self,action : Action):
        # action is ROTATION @ TRANSLATION, SO A 2X2 matrix 
        # we need to update self.x, self.y and self.angle respectively
        
        state_change_action = action.state_change
        angle = action.orientation
        # Update the object's angle (add the rotation to current angle)
        self.angle = angle
        
        # Optional: Keep angle in [0, 360) range
        self.angle = self.angle % 360
        # action.rotation = self.angle # have to hard reset. need to change st actions now represent absolute final position 
        moving_action = action.movement_as_matrix()

        # Update position
        self.x += moving_action[0]
        self.y += moving_action[1]

        if state_change_action is not None and state_change_action != self.state: #here state change action will be represented by PlayerState
            self.alternate_state()
             
    # TL, TR, BR, BL, center dictonary
    def get_pos(self):

        return {
            'top-left': (self.x - self.size, self.y - self.size),
            'top-right' : (self.x + self.size, self.y - self.size),
            'bot-right' : (self.x + self.size, self.y + self.size),
            'bot-left' : (self.x - self.size, self.y + self.size),
            'center' : (self.x, self.y),
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
            'center': (self.x, self.y),
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
            'center': (self.x, self.y),
        }
    
class Goal:
    def __init__(self, x, y, width=50, height=50):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                          self.width, self.height)
    
    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, 
                        (self.x - 25, self.y - 25, 50, 50))
        pygame.draw.rect(screen, BLACK, 
                        (self.x - 25, self.y - 25, 50, 50), 3)
    
    # TL, TR, BR, BL, Center dictonary
    def get_pos(self):
        return {
            'top-left': (self.x - self.width//2, self.y - self.height//2),
            'top-right': (self.x + self.width//2, self.y - self.height//2),
            'bot-right': (self.x + self.width//2, self.y + self.height//2),
            'bot-left': (self.x - self.width//2, self.y + self.height//2),
            'center': (self.x, self.y),
        }

class GameObjective(Enum):
    EAT_ALL = 1
    REACH_GOAL = 2

    
 
class Game:
    
    def __init__(self, num_edibles = NUM_OF_EDIBLE_OBJECT, num_obstacles = NUM_OF_OBSTACLES, num_goals = NUM_OF_GOALS, screen_width = SCREEN_WIDTH, screen_height = SCREEN_HEIGHT, objective = None):
        self.num_edibles = num_edibles
        self.num_obstacles = num_obstacles
        self.num_goals = num_goals
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("2D Game - Eat or Avoid")
        self.clock = pygame.time.Clock()
        
        # Initialize game objects
        self.player = Player(100, 100)
        self.edibles = []
        self.obstacles = []
        self.goal = None
        
        # Game state
        self.objective = objective  # "eat_all" or "reach_goal"
        self.game_over = False
        self.game_won = False
        self.font = pygame.font.Font(None, 36)

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

    def handle_action(self,action : Action):
  
        if self.game_over or self.game_won:
            self.restart_game()
        
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
        self.player = Player(100, 100)
        self.edibles = []
        self.obstacles = []
        self.goal = None
        self.game_over = False
        self.game_won = False
        self.setup_game()
    
    def end_game(self):
        pygame.quit()
        sys.exit()

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
            [ np.array(self.screen.get_at((x,y))[:3]) for y in range(self.screen_height)] for x in range(self.screen_width)
        ]

        return np.array(pixels)


# NEED TO INCORPORATE GAMEINTERFACE SUCH THAT IT CAN TAKE IN DEMO AND AGENT MOVES 
class GameMode(Enum):
    DEMO_MODE = 1
    AGENT_MODE = 2 


# make GameInteraface s.t an action can be done with Translation + Rotation matrix
class GameInterface:

    def __init__(self,num_edibles = NUM_OF_EDIBLE_OBJECT, num_obstacles = NUM_OF_OBSTACLES, num_sampled_points = 10, mode = GameMode.DEMO_MODE):
        self.game = Game(num_edibles=num_edibles, num_obstacles=num_obstacles)
        self.running = True
        self.t = 0
        self.num_sampled_points = num_sampled_points
        self.mode = mode

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
        raw_coords = np.array([[(x,y) for y in range(self.game.screen_height) ]  for x in range(self.game.screen_width)])

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
            'agent-orientation' : self.game.player.angle,
            'agent-state' : agent_state,
            'done': self.running,
            'time' : self.t
        }

    def step(self,action = None):
        if self.running:
            self.t += 1
            if self.mode == GameMode.DEMO_MODE:
                self.running = self.game.handle_events()
            else:
                assert action is not None
                self.running = self.game.handle_action(action) 
                
            self.game.update()
            self.game.draw()
            self.game.clock.tick(60)
            obs = self.get_obs()
            return obs 
        else:
            pygame.quit()
            sys.exit()
    
    def _get_agent_pos(self):
        player_pos = self.game.player.get_pos()
        tl = np.array(player_pos['top-left'])
        tr = np.array(player_pos['top-right'])
        br = np.array(player_pos['bot-right'])
        bl = np.array(player_pos['bot-left'])
        center = np.array(player_pos['center'])
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

"""
    # return dictionary of point clouds taken from screen
    # for now we just return the corners and centroid of each object
    # each point cloud encodes the position and colour of the particular pixel 
    # belows will be implemented in another function by class Subgraph and Graph within the agent file 
        # the collected point clouds are used to construct a sub-graph which represents the object geometry/shape 
            # this edge type will be intra-object edge-type, and its feature also includes a distance based on euc-distance for now
        # then to construct the local graph, corners of each object are fully connected to all other corners of other object, and centroid are connected to other centroids
            # this edge type will be inter-object edge-type, and its feature also includes a distance based on euc-distance for now
    # def _get_game_obj_pos(self):
    #     # first get player 
    #     player_pos = self.game.player.get_pos()

    #     # then get obstacles and object 
    #     obstacles = [obj.get_pos() for obj in self.game.obstacles]
    #     edibles = [edi.get_pos() for edi in self.game.edibles]
    #     objs_pos = obstacles + edibles

    #     return {
    #         'player' : player_pos,
    #         'objects' : objs_pos,
    #         'done' : self.running,
    #         'objective' : self.game.objective
    #     }
"""

# this class is used to represent a pseudogame
# It should play out by itself 
# for pseudo demonstrations we will only concern ourselves with 1 object 

class PseudoGameMode:
    EAT_EDIBLE = 1
    AVOID_OBSTACLE = 2
    REACH_GOAL = 3
    RANDOM = 4 



# pseudogame used to generate a pseudo demo
class PseudoGame:

    def __init__(self, max_num_sampled_waypoints = 6, min_num_sampled_waypoints = 2, mode = PseudoGameMode.RANDOM, biased = False, 
                 augmented = False, screen_width = 400, screen_height = 300, num_sampled_point_clouds = 20):
        
        # setup configs for pseudo game
        self.mode = mode 
        self.num_sampled_point_clouds = num_sampled_point_clouds
        self._init_game_configs()

        # init game
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.player = Player(100, 100)
        self.t = 0
        self.temp = 0 # remove this when useless
        self.object = None

        # setup game by creating relevant object
        self._setup_pseudo_game()

        self.num_waypoints_used = np.random.randint(min_num_sampled_waypoints, max_num_sampled_waypoints)
        self.waypoints = None 
        self.biased = biased
        self.augmented = augmented
        self.waypoints = []
        self.observations = []
        self._sample_waypoints()


    # this function does a run of the pseudogame and returns the observations in pointclouds 
    # it will be a list of 'frames' at each timepoints
    def run(self):
        self.draw()
        obs = self.get_obs()
        self.observations.append(obs)
        while self.num_waypoints_used > self.t:
            self.go_to_next_waypoint()
            self.update()
            self.draw()
            self.clock.tick(60)
            self.t += 1
            obs = self.get_obs()
            self.observations.append(obs)
        self.update()
        self.draw()
        self.clock.tick(60)
        self._end_game()


    def get_obs(self):
        agent_pos = self._get_agent_pos()
        # Since our 'point clouds' are represented as pixels in a 2d grid, our dense point cloud will be a 2d matrix of Screen-width x Screen-height
        raw_dense_point_clouds = self._get_screen_pixels()
        raw_coords = np.array([[(x,y) for y in range(self.screen_height) ]  for x in range(self.screen_width)])

        # To ensure that only objects point clouds are picked up, we remove all white pixels and agent pixels
        mask = ~np.all((raw_dense_point_clouds == [255, 255, 255]) | 
                    (raw_dense_point_clouds == [128, 0, 128]) |
                    (raw_dense_point_clouds == [0, 0, 255]) |
                    (raw_dense_point_clouds == [125, 125, 125]) 
                    , axis=2)
        valid_points = np.where(mask)
        dense_point_clouds = raw_dense_point_clouds[valid_points]
        coords = raw_coords[valid_points]

        # To simulate FPSA, we randomly select an initial point from coords ranging from p in (SCREEN_WIDTH X SCREEN_HEIGHT)  
        selected_indices = []
        initial_coord = random.randint(0, len(coords) - 1)
        selected_indices.append(initial_coord)

        # then perform FPSA to collect M points
        for _ in range(self.num_sampled_point_clouds-1):
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
            'agent-orientation' : self.player.angle,
            # 'done': self.running,
            # 'time' : self.t
        }


    # needs to change too since we are switching x and y
    def go_to_next_waypoint(self):

        if len(self.waypoints) == 0:
            return
        next_waypoint = self.waypoints.pop(0)

        next_point = next_waypoint['movement']
        state_change = next_waypoint['state-change']
        player_pos = self.player.get_pos()['center']
        # deal with movement first
        # we default movement to be a tuple of (distance, angle)

        if next_point is not None:
            dydx =  next_point - player_pos # 
            curr_player_angle = self.player.angle
            # angle_rad = np.arctan2(dydx[0], dydx[1])  # assuming [dy, dx]
            angle_rad = np.arctan2(dydx[1], dydx[0])  # assuming [dy, dx]
            final_angle_deg = np.degrees(angle_rad)
            
            # then find distance
            distance = np.linalg.norm(dydx)
        print(f"trans : {distance}, rot : {final_angle_deg}")
        action = Action(forward_movement=distance, orientation=final_angle_deg, state_change=state_change) 
        self.player.move_with_action(action)


    def update(self):
        player_rect = self.player.get_rect()
        object_rect = self.object.get_rect()
        if player_rect.colliderect(object_rect) and type(self.object) == EdibleObject and self.player.state is PlayerState.EATING:
            self.object.eaten = True 

    def draw(self):
        self.screen.fill(WHITE)
        
        self.player.draw(self.screen)
        self.object.draw(self.screen)
        for waypoint in self.waypoints:
            center = waypoint['movement']
            pygame.draw.circle(self.screen, (125, 125, 125), center, 5)
        pygame.display.flip()

    def _get_agent_pos(self):
        player_pos = self.player.get_pos()
        tl = np.array(player_pos['top-left'])
        tr = np.array(player_pos['top-right'])
        br = np.array(player_pos['bot-right'])
        bl = np.array(player_pos['bot-left'])
        center = np.array(player_pos['center'])
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
    
    def _init_game_configs(self):
        if self.mode == PseudoGameMode.RANDOM:
            self.mode = random.choice([PseudoGameMode.EAT_EDIBLE, PseudoGameMode.AVOID_OBSTACLE, PseudoGameMode.REACH_GOAL])
        

    def _setup_pseudo_game(self):
        if self.mode == PseudoGameMode.EAT_EDIBLE:
            x = random.randint(50, self.screen_width - 50)
            y = random.randint(50, self.screen_height - 50)
            # Make sure edibles don't spawn too close to player
            while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 100:
                x = random.randint(50, self.screen_width - 50)
                y = random.randint(50, self.screen_height - 50)
            
            width = random.randint(15, 25)
            height = random.randint(15, 25)
            self.object = EdibleObject(x, y, width, height)

        elif self.mode == PseudoGameMode.AVOID_OBSTACLE:
            x = random.randint(50, self.screen_width - 50)
            y = random.randint(50, self.screen_height - 50)
            # Make sure obstacles don't spawn too close to player
            while math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2) < 150:
                x = random.randint(50,  self.screen_width - 50)
                y = random.randint(50, self.screen_height - 50)
            
            width = random.randint(30, 50)
            height = random.randint(30, 50)
            self.object = Obstacle(x, y, width, height)

        else:
            x = random.randint(50, self.screen_width - 50)
            y = random.randint(50, self.screen_height - 50)
            self.object = Goal(x,y)
        
    # this function is very bad
    # for edibles and goal we need it to actually move towards it
    # but not doing it
    def _sample_waypoints(self):
        game_obj_positions = self._get_game_obj_pos()
        object_pos = game_obj_positions['object']  # tl, tr, br, bl, center
        agent_pos = game_obj_positions['player']  # tl, tr, br, bl, center
        waypoints = []


        if type(self.object) is EdibleObject or type(self.object) is Goal:
            # want to sample waypoints near object s.t it cross
            # for random we sample random waypoints
            # for biased, we sample waypoints along vector from center of player to center of goal
            if self.biased:

                # Sample waypoints along the direct path to the goal
                player_center = agent_pos['center']
                object_center = object_pos['center']
                direction_vector = np.array(object_center) - np.array(player_center)
                for i in range(1, self.num_waypoints_used + 1):
                    # Sample points along the line with some random offset
                    t = i / (self.num_waypoints_used + 1)  # Parameter along the line
                    base_point = np.array(player_center) + t * direction_vector
                    
                    # Add small random offset perpendicular to the direction
                    perpendicular = np.array([-direction_vector[1], direction_vector[0]])
                    perpendicular = perpendicular / np.linalg.norm(perpendicular) if np.linalg.norm(perpendicular) > 0 else np.array([0, 0])
                    offset = np.random.uniform(-20, 20) * perpendicular
                    
                    waypoint = base_point + offset
                    waypoints.append(waypoint)

            else:

                object_center = object_pos['center']
                obj_width = self.object.width
                obj_height = self.object.height
                min_radius = min( obj_width, obj_height ) + self.player.size + 5

                for _ in range(self.num_waypoints_used):
                    # Sample random points in a circle around the object
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(min_radius, min_radius * 1.5)  # Adjust radius as needed
                    offset = radius * np.array([np.cos(angle), np.sin(angle)])
                    waypoint = np.array(object_center) + offset
                    waypoints.append(waypoint)

        else:
            # for random, sample waypoints away from object
            # for biased, sample waypooints s.t it circumvents the obstacle
            # this is done by sampling points around tl, tr, br, bl of the obstacle (if there is enough waypoints)
            if self.biased:
                # Sample waypoints around the corners to circumvent the obstacle
                corners = [object_pos['top-left'], object_pos['top-right'], object_pos['bot-right'], object_pos['bot-left']]
                player_center = agent_pos['center']
                object_center = object_pos['center']
                
                # Determine which side of the obstacle to go around
                player_to_object = np.array(object_center) - np.array(player_center)
                
                waypoints_per_corner = max(1, self.num_waypoints_used // 4 + 1)
                
                for corner in corners:
                    for i in range(waypoints_per_corner):
                        # Sample points around each corner with some offset
                        corner_array = np.array(corner)
                        
                        # Offset away from the obstacle center
                        corner_to_center = corner_array - np.array(object_center)
                        corner_to_center = corner_to_center / np.linalg.norm(corner_to_center) if np.linalg.norm(corner_to_center) > 0 else np.array([1, 0])
                        
                        # Add random offset in the direction away from obstacle
                        offset_distance = np.random.uniform(20, 60)
                        random_angle = np.random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
                        
                        rotation_matrix = np.array([[np.cos(random_angle), -np.sin(random_angle)],
                                                [np.sin(random_angle), np.cos(random_angle)]])
                        offset_direction = rotation_matrix @ corner_to_center
                        
                        waypoint = corner_array + offset_distance * offset_direction
                        waypoints.append(waypoint)
                        
                        if len(waypoints) >= self.num_waypoints_used:
                            break
                    if len(waypoints) >= self.num_waypoints_used:
                        break
                        
                # Trim to exact number needed
                waypoints = waypoints[:self.num_waypoints_used]
            else:
                # Random waypoints away from the obstacle
                object_center = object_pos['center']
                player_center = agent_pos['center']
                
                for _ in range(self.num_waypoints_used):
                    # Sample random points, but bias them away from the obstacle
                    attempts = 0
                    while attempts < 10:  # Limit attempts to avoid infinite loop
                        # Generate random point
                        angle = np.random.uniform(0, 2 * np.pi)
                        radius = np.random.uniform(30, 100)  # Distance from player
                        candidate = np.array(player_center) + radius * np.array([np.cos(angle), np.sin(angle)])
                        
                        # Check if point is reasonably far from obstacle
                        distance_to_obstacle = np.linalg.norm(candidate - np.array(object_center))
                        if distance_to_obstacle > 40:  # Minimum distance from obstacle
                            waypoints.append(candidate)
                            break
                        attempts += 1
                    
                    # If we couldn't find a good point, just add a random one
                    if attempts >= 10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        radius = np.random.uniform(50, 120)
                        waypoint = np.array(player_center) + radius * np.array([np.cos(angle), np.sin(angle)])
                        waypoints.append(waypoint)

        # now we got waypoints, we need to assign 1 or more waypoints that get chosen to alter state
        processed_waypoints = [{'movement' : waypoint, 
                                'state-change' : None} 
                                for waypoint in waypoints]
        num_state_changing_waypoints = np.random.randint(1,self.num_waypoints_used)
        chosen_index = np.random.choice(self.num_waypoints_used, replace = False, size = num_state_changing_waypoints)
        state_change_to = PlayerState.EATING

        # now choose 1 or more to alternate states
        for index in chosen_index:
            processed_waypoints[index]['state-change'] = state_change_to
            if state_change_to is PlayerState.EATING:
                state_change_to = PlayerState.NOT_EATING
            else:
                state_change_to = PlayerState.EATING


        # for now we dont augment
        if self.augmented:



            pass
        pass
        self.waypoints = processed_waypoints
        
    def _get_game_obj_pos(self):
        # first get player 
        player_pos = self.player.get_pos()
        object_pos = self.object.get_pos()

        return {
            'player' : player_pos,
            'object' : object_pos
        }

    def _get_screen_pixels(self):
        pixels = [
            [ np.array(self.screen.get_at((x,y))[:3]) for y in range(self.screen_height)] for x in range(self.screen_width)
        ]

        return np.array(pixels)
    
    def _end_game(self):
        pygame.quit()
        sys.exit()
    


# Run the game
if __name__ == "__main__":

    # Code to try Game
    # game = Game()
    # game.run()

    # Code to try Game Interface for agent
    # game_interface = GameInterface()
    # game_interface.start_game()
    # game_interface.step()
    
    # Code to try PseudoGame
    pseudogame = PseudoGame(biased=False)
    # sample points
    pseudogame.waypoints = [{'movement' : np.array([150,125]), 
                                'state-change' : None} ,
                                {'movement' :  np.array([175,200]), 
                                'state-change' : None} ,
                                {'movement' :  np.array([225,250]), 
                                'state-change' : None} ,
                                {'movement' :  np.array([300,275]), 
                                'state-change' : None} ,]
                                # {'movement' :  np.array([250,200]), 
                                # 'state-change' : None} ]
    pseudogame.num_waypoints_used = 5
    pseudogame.run()
    # print('hi')
    # print(pseudogame.observations)
    pass 