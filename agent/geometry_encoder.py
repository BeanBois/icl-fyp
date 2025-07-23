
import random 
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

def FPSA(dense_point_clouds, coords, num_sampled_points):
    selected_indices = []
    initial_coord = random.randint(0, len(coords) - 1)
    selected_indices.append(initial_coord)
    from tasks.twoD.game import BLUE, PURPLE, BLACK, YELLOW, GREEN, RED
    # then perform FPSA to collect M points
    for _ in range(num_sampled_points-1):
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

        # fix here
        return {
            'point-clouds': dict(color_segments),
            'agent-pos' : agent_pos,
            'agent-orientation' : self.game.player.angle,
            'agent-state' : agent_state,
            'agent-keypoints' : self.player.get_keypoints(),
            'done': self.running,
            'time' : self.t
        }

    pass 

# the SA layers has 3 components 
    # Sampling layer: FPSA (should move FPSA here)
    # Grouping Layer: 
class SetAbstractionLayer(nn.Module):

    def __init__(self, dim_size):
        super(SetAbstractionLayer, self).__init__()

        self.sampling_layer = FPSA

        pass 

    def forward(self, pcs):
        pass 




       