from .game import Game, GameMode
import numpy as np
import os 
from .game_configs import NUM_OF_EDIBLE_OBJECT, NUM_OF_OBSTACLES, BLACK, YELLOW, RED, GREEN
from collections import defaultdict

class GameInterface:

    def __init__(self,num_edibles = NUM_OF_EDIBLE_OBJECT, num_obstacles = NUM_OF_OBSTACLES, num_sampled_points = 10, mode = GameMode.DEMO_MODE):
        # dont run headless
        if 'SDL_VIDEODRIVER' in os.environ:
            del os.environ['SDL_VIDEODRIVER']

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
  
    def reset(self):
        self.game.restart_game()
        self.running = True
        self.t = 0

    def change_mode(self,mode):
        self.mode = mode
    # we do furthest point sampling algorithm to retrieve a cloud of point clouds
    # These sampled ppoint clouds is then used to create a graph representation 

    # remove FPSA form here and just return pc
    def get_obs(self):
        agent_pos = self._get_agent_pos()

        agent_state = self.game.player.state # works now but maybe refactor

        # Since our 'point clouds' are represented as pixels in a 2d grid, our dense point cloud will be a 2d matrix of Screen-width x Screen-height
        raw_dense_point_clouds = self.game.get_screen_pixels()
        raw_coords = np.array([[(x,y) for y in range(self.game.screen_height) ]  for x in range(self.game.screen_width)])
        
        # To ensure that only objects point clouds are picked up, we remove all white pixels and agent pixels
        mask = ~np.all((raw_dense_point_clouds == [255, 255, 255]) | 
                    (raw_dense_point_clouds == [128, 0, 128]) |
                    (raw_dense_point_clouds == [0, 0, 255]) |
                    (raw_dense_point_clouds != [128, 0 ,128]) | # ignore agent pc
                    (raw_dense_point_clouds != [0, 0 ,255])  # ignore agent pc
                    , axis=2)
        

        valid_points = np.where(mask)
        dense_point_clouds = raw_dense_point_clouds[valid_points]
        coords = raw_coords[valid_points]

        return {
            #segmented point clouds 
            'point-clouds': dense_point_clouds,
            'coords' : coords,
            # agent info
            'agent-pos' : agent_pos,
            'agent-state' : agent_state,
            'agent-orientation' : self.game.player.get_orientation('deg'),
            #game info
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
    
    def _get_agent_pos(self):
        _, front, back_left, back_right = [v for _, v in self.game.player.get_keypoints(frame = 'self').items()]
        center = self.game.player.get_pos()
        center = np.array(center)
        # player is a triangle so i want to capture the 3 edges of the triangle
        # at player_ori == 0 degree, edges == (tl, bl, (tr+br)//2)
        tri_points = np.array([front, back_left, back_right])
        ori_in_deg = self.game.player.get_orientation(mode='deg')
        R = self._rotation_matrix_2d(ori_in_deg) 

        # rotate around center

        rotated = (R @ tri_points.T).T
        final = rotated + center
        player_pos = np.vstack([center, final])
        return player_pos   
    
    def _rotation_matrix_2d(self, theta_deg):
        theta = theta_deg/180 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s],
                        [s,  c]])