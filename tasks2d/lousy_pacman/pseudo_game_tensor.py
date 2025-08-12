import torch
import numpy as np
import math
import random
from typing import List, Tuple, Optional
from .pseudo_game_aux import *
from .pseudo_game import PseudoGame



def batch_obstacle_blocking(player_pos: torch.Tensor, goal_center: torch.Tensor, 
                          obst_center: torch.Tensor, obst_width: torch.Tensor, 
                          obst_height: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized obstacle blocking detection for batch processing.
    
    Args:
        player_pos: (batch_size, 2) - player positions
        goal_center: (batch_size, 2) - goal centers
        obst_center: (batch_size, 2) - obstacle centers
        obst_width: (batch_size,) - obstacle widths
        obst_height: (batch_size,) - obstacle heights
    
    Returns:
        is_blocking: (batch_size,) - boolean tensor
        closest_points: (batch_size, 2) - closest points on line
    """
    batch_size = player_pos.shape[0]
    device = player_pos.device
    
    # Vector from player to goal
    to_goal = goal_center - player_pos
    to_goal_norm = torch.norm(to_goal, dim=1, keepdim=True)
    to_goal_normalized = to_goal / (to_goal_norm + 1e-8)  # avoid division by zero
    
    # Vector from player to obstacle center
    to_obstacle = obst_center - player_pos
    
    # Project obstacle center onto the line from player to goal
    projection_length = torch.sum(to_obstacle * to_goal_normalized, dim=1)
    
    # Check if projection is negative or beyond goal
    invalid_projection = (projection_length < 0) | (projection_length > to_goal_norm.squeeze(1))
    
    # Find the closest point on the line to the obstacle center
    closest_points = player_pos + projection_length.unsqueeze(1) * to_goal_normalized
    
    # Calculate distance from obstacle center to the line
    distance_to_line = torch.norm(obst_center - closest_points, dim=1)
    
    # Check if obstacle intersects with the path
    obstacle_radius = torch.max(obst_width, obst_height) / 2
    is_intersecting = distance_to_line < obstacle_radius
    
    # Combine conditions
    is_blocking = is_intersecting & (~invalid_projection)
    
    return is_blocking, closest_points

def batch_rotation_matrix_2d(theta_deg: torch.Tensor) -> torch.Tensor:
    """
    Create batch of 2D rotation matrices.
    
    Args:
        theta_deg: (batch_size,) - rotation angles in degrees
    
    Returns:
        rotation_matrices: (batch_size, 2, 2)
    """
    theta_rad = theta_deg * torch.pi / 180.0
    cos_theta = torch.cos(theta_rad)
    sin_theta = torch.sin(theta_rad)
    
    batch_size = theta_deg.shape[0]
    R = torch.zeros(batch_size, 2, 2, device=theta_deg.device)
    R[:, 0, 0] = cos_theta
    R[:, 0, 1] = -sin_theta
    R[:, 1, 0] = sin_theta
    R[:, 1, 1] = cos_theta
    
    return R

def batch_interpolate_waypoints(waypoints: torch.Tensor, target_num_waypoints: int, 
                               noise_mean: float = 0.0, noise_std: float = 1.0) -> torch.Tensor:
    """
    Batch interpolation of waypoints to reach target number.
    
    Args:
        waypoints: (batch_size, num_waypoints, waypoint_dim)
        target_num_waypoints: desired number of waypoints
        noise_mean: mean for interpolation noise
        noise_std: std for interpolation noise
    
    Returns:
        interpolated_waypoints: (batch_size, target_num_waypoints, waypoint_dim)
    """
    batch_size, current_num, waypoint_dim = waypoints.shape
    device = waypoints.device
    
    if current_num >= target_num_waypoints:
        return waypoints[:, :target_num_waypoints]
    
    # Calculate how many waypoints to add
    num_to_add = target_num_waypoints - current_num
    
    # For each batch, randomly select segments to interpolate
    all_interpolated = []
    
    for b in range(batch_size):
        current_waypoints = waypoints[b]
        interpolated = [current_waypoints]
        
        for _ in range(num_to_add):
            if current_waypoints.shape[0] < 2:
                break
                
            # Randomly choose a segment
            segment_idx = torch.randint(0, current_waypoints.shape[0] - 1, (1,)).item()
            
            # Interpolate between waypoints
            point_a = current_waypoints[segment_idx]
            point_b = current_waypoints[segment_idx + 1]
            midpoint = (point_a + point_b) / 2
            
            # Add noise
            if waypoint_dim > 1:  # Only add noise to position coordinates
                noise = torch.normal(noise_mean, noise_std, (waypoint_dim - 1,), device=device)
                midpoint[:-1] += noise
                
            # Insert the new waypoint
            current_waypoints = torch.cat([
                current_waypoints[:segment_idx + 1],
                midpoint.unsqueeze(0),
                current_waypoints[segment_idx + 1:]
            ], dim=0)
        
        # Pad or trim to exact target length
        if current_waypoints.shape[0] < target_num_waypoints:
            padding = torch.zeros(target_num_waypoints - current_waypoints.shape[0], waypoint_dim, device=device)
            current_waypoints = torch.cat([current_waypoints, padding], dim=0)
        else:
            current_waypoints = current_waypoints[:target_num_waypoints]
            
        all_interpolated.append(current_waypoints)
    
    return torch.stack(all_interpolated)

class TensorizedPseudoGame:
    agent_keypoints = PseudoGame.agent_keypoints
    """Tensorized version of PseudoGame for efficient batch processing."""
    
    def __init__(self, batch_size: int = 32, device: str = 'cpu', **kwargs):
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Transfer relevant parameters from original class
        self.num_objects = kwargs.get('num_objects', NUM_OF_OBJECTS)
        self.max_num_sampled_waypoints = kwargs.get('max_num_sampled_waypoints', MAX_NUM_SAMPLED_WAYPOINTS)
        self.min_num_sampled_waypoints = kwargs.get('min_num_sampled_waypoints', MIN_NUM_SAMPLED_WAYPOINTS)
        self.screen_width = kwargs.get('screen_width', SCREEN_WIDTH)
        self.screen_height = kwargs.get('screen_height', SCREEN_HEIGHT)
        self.biased = kwargs.get('biased', DEFAULT_NOT_BIASED)
        self.augmented = kwargs.get('augmented', DEFAULT_NOT_AUGMENTED)
        self.max_length = kwargs.get('MAX_LENGTH', 100)
        
    def generate_batch_trajectories(self) -> dict:
        """Generate a batch of trajectories efficiently using tensor operations."""
        
        # Initialize batch tensors
        batch_waypoints = []
        batch_actions = []
        batch_observations = []
        
        # Generate object configurations for the batch
        batch_object_configs = self._generate_batch_object_configs()
        
        # Generate waypoints for the batch
        if self.biased:
            batch_waypoints = self._generate_biased_waypoints_batch(batch_object_configs)
        else:
            batch_waypoints = self._generate_random_waypoints_batch(batch_object_configs)
        
        # Apply augmentation if enabled
        if self.augmented:
            batch_waypoints = self._augment_waypoints_batch(batch_waypoints)
        
        # Generate actions from waypoints
        batch_actions = self._generate_actions_from_waypoints_batch(batch_waypoints, batch_object_configs)
        
        # Generate observations (simplified - you may need to adapt this)
        batch_observations = self._generate_observations_batch(batch_object_configs, batch_waypoints)
        
        return {
            'waypoints': batch_waypoints,
            'actions': batch_actions,
            'observations': batch_observations,
            'object_configs': batch_object_configs
        }
    
    def _generate_batch_object_configs(self) -> dict:
        """Generate object configurations for the entire batch."""
        batch_configs = {
            'player_positions': torch.randint(0, min(self.screen_width, self.screen_height), 
                                            (self.batch_size, 2), device=self.device).float(),
            'player_orientations': torch.rand(self.batch_size, device=self.device) * 360,
            'object_positions': torch.randint(0, min(self.screen_width, self.screen_height), 
                                            (self.batch_size, self.num_objects, 2), device=self.device).float(),
            'object_types': torch.randint(0, len(AVAILABLE_OBJECTS), 
                                        (self.batch_size, self.num_objects), device=self.device),
            'object_dimensions': torch.rand(self.batch_size, self.num_objects, 2, device=self.device) * 60 + 30  # random width/height
        }
        return batch_configs
    
    def _generate_biased_waypoints_batch(self, batch_configs: dict) -> torch.Tensor:
        """Generate biased waypoints for the entire batch using tensor operations."""
        batch_waypoints = []
        
        for b in range(self.batch_size):
            # Get configurations for this batch element
            player_pos = batch_configs['player_positions'][b]
            object_positions = batch_configs['object_positions'][b]
            object_types = batch_configs['object_types'][b]
            object_dims = batch_configs['object_dimensions'][b]
            
            # Determine scenario based on object types (simplified logic)
            num_waypoints = torch.randint(self.min_num_sampled_waypoints, 
                                        self.max_num_sampled_waypoints + 1, (1,)).item()
            
            # Create basic waypoint path (can be extended with more sophisticated logic)
            waypoints = self._create_waypoint_path(player_pos, object_positions, object_types, 
                                                 object_dims, num_waypoints)
            batch_waypoints.append(waypoints)
        
        # Pad waypoints to same length for batching
        max_waypoints = max(wp.shape[0] for wp in batch_waypoints)
        padded_waypoints = []
        
        for waypoints in batch_waypoints:
            if waypoints.shape[0] < max_waypoints:
                padding = torch.zeros(max_waypoints - waypoints.shape[0], waypoints.shape[1], 
                                    device=self.device)
                waypoints = torch.cat([waypoints, padding], dim=0)
            padded_waypoints.append(waypoints)
        
        return torch.stack(padded_waypoints)
    
    def _create_waypoint_path(self, player_pos: torch.Tensor, object_positions: torch.Tensor,
                            object_types: torch.Tensor, object_dims: torch.Tensor, 
                            num_waypoints: int) -> torch.Tensor:
        """Create a waypoint path for a single trajectory."""
        waypoints = [player_pos]
        
        # Simple strategy: go to first object of interest
        for i, obj_type in enumerate(object_types):
            if obj_type == 0:  # Assuming 0 is edible object
                target_pos = object_positions[i]
                waypoints.append(target_pos)
                break
        
        # Add more waypoints through interpolation if needed
        while len(waypoints) < num_waypoints + 1:
            if len(waypoints) < 2:
                break
            # Insert waypoint between random adjacent points
            idx = torch.randint(0, len(waypoints) - 1, (1,)).item()
            midpoint = (waypoints[idx] + waypoints[idx + 1]) / 2
            # Add small noise
            noise = torch.normal(0, 2.0, (2,), device=self.device)
            midpoint += noise
            waypoints.insert(idx + 1, midpoint)
        
        # Remove initial position and create waypoint tensor with state
        waypoints = waypoints[1:]
        waypoint_dim = 3  # x, y, state
        waypoints_tensor = torch.zeros(len(waypoints), waypoint_dim, device=self.device)
        
        for i, wp in enumerate(waypoints):
            waypoints_tensor[i, :2] = wp
            waypoints_tensor[i, 2] = PlayerState.NOT_EATING.value  # Default state
        
        return waypoints_tensor
    
    def _generate_random_waypoints_batch(self, batch_configs: dict) -> torch.Tensor:
        """Generate random waypoints for the entire batch."""
        batch_waypoints = []
        
        for b in range(self.batch_size):
            num_waypoints = torch.randint(self.min_num_sampled_waypoints, 
                                        self.max_num_sampled_waypoints + 1, (1,)).item()
            
            # Generate random waypoints
            waypoints = torch.rand(num_waypoints, 3, device=self.device)
            waypoints[:, 0] *= self.screen_width  # x coordinates
            waypoints[:, 1] *= self.screen_height  # y coordinates
            waypoints[:, 2] = torch.randint(0, 2, (num_waypoints,), device=self.device)  # states
            
            batch_waypoints.append(waypoints)
        
        # Pad to same length
        max_waypoints = max(wp.shape[0] for wp in batch_waypoints)
        padded_waypoints = []
        
        for waypoints in batch_waypoints:
            if waypoints.shape[0] < max_waypoints:
                padding = torch.zeros(max_waypoints - waypoints.shape[0], 3, device=self.device)
                waypoints = torch.cat([waypoints, padding], dim=0)
            padded_waypoints.append(waypoints)
        
        return torch.stack(padded_waypoints)
    
    def _augment_waypoints_batch(self, batch_waypoints: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to batch of waypoints."""
        batch_size, num_waypoints, waypoint_dim = batch_waypoints.shape
        augmented = batch_waypoints.clone()
        
        # Apply local disturbances to 30% of trajectories
        disturbance_mask = torch.rand(batch_size, device=self.device) < 0.3
        
        if disturbance_mask.any():
            # Add noise to position coordinates
            noise = torch.normal(0, 15.0, (batch_size, num_waypoints, 2), device=self.device)
            augmented[disturbance_mask, :, :2] += noise[disturbance_mask]
            
            # Clamp to bounds
            augmented[:, :, 0] = torch.clamp(augmented[:, :, 0], 0, self.screen_width - 1)
            augmented[:, :, 1] = torch.clamp(augmented[:, :, 1], 0, self.screen_height - 1)
        
        # Change states for 10% of data points
        state_change_mask = torch.rand(batch_size, num_waypoints, device=self.device) < 0.1
        augmented[:, :, 2][state_change_mask] = 1 - augmented[:, :, 2][state_change_mask]
        
        return augmented
    
    def _generate_actions_from_waypoints_batch(self, batch_waypoints: torch.Tensor, 
                                             batch_configs: dict) -> torch.Tensor:
        """Generate actions from waypoints for the entire batch."""
        batch_size, num_waypoints, _ = batch_waypoints.shape
        
        # Initialize action tensor (forward_movement, rotation, state_change)
        batch_actions = torch.zeros(batch_size, num_waypoints, 3, device=self.device)
        
        player_positions = batch_configs['player_positions']
        player_orientations = batch_configs['player_orientations']
        
        for t in range(num_waypoints):
            if t == 0:
                current_positions = player_positions
                current_orientations = player_orientations
            else:
                current_positions = batch_waypoints[:, t-1, :2]
                # Update orientations based on previous actions (simplified)
                current_orientations = current_orientations + batch_actions[:, t-1, 1]
            
            target_positions = batch_waypoints[:, t, :2]
            target_states = batch_waypoints[:, t, 2]
            
            # Calculate movement vectors
            dydx = target_positions - current_positions
            distances = torch.norm(dydx, dim=1)
            
            # Calculate angles
            angles_rad = torch.atan2(dydx[:, 1], dydx[:, 0])
            angles_deg = angles_rad * 180 / torch.pi
            
            # Calculate rotation needed
            rotation_needed = angles_deg - current_orientations
            
            # Normalize rotation to [-180, 180]
            rotation_needed = torch.remainder(rotation_needed + 180, 360) - 180
            
            # Clamp actions
            forward_movement = torch.clamp(distances, 0, MAX_FORWARD_DIST)
            rotation = torch.clamp(rotation_needed, -MAX_ROTATION, MAX_ROTATION)
            
            batch_actions[:, t, 0] = forward_movement
            batch_actions[:, t, 1] = rotation
            batch_actions[:, t, 2] = target_states
        
        return batch_actions
    
    def _get_agent_pos_batch(self, player_positions: torch.Tensor, player_orientations: torch.Tensor) -> torch.Tensor:
        """
        Batch version of _get_agent_pos that creates triangle keypoints for each player.
        
        Args:
            player_positions: (batch_size, 2) - player center positions
            player_orientations: (batch_size,) - player orientations in degrees
            
        Returns:
            agent_pos_batch: (batch_size, 4, 2) - center + 3 triangle points for each player
        """
        batch_size = player_positions.shape[0]
        device = player_positions.device
        
        # Get triangle keypoints relative to player (same as PseudoPlayer.get_keypoints)
        # These are the relative positions when orientation = 0
        front = torch.tensor([0., -10.], device=device)  # Front point
        back_left = torch.tensor([-8., 6.], device=device)  # Back left
        back_right = torch.tensor([8., 6.], device=device)  # Back right
        
        # Stack triangle points: (3, 2)
        tri_points = torch.stack([front, back_left, back_right], dim=0)
        
        # Create rotation matrices for each player: (batch_size, 2, 2)
        R_batch = batch_rotation_matrix_2d(player_orientations)
        
        # Rotate triangle points for each player
        # tri_points: (3, 2) -> (1, 3, 2) -> (batch_size, 3, 2)
        tri_points_expanded = tri_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply rotation: (batch_size, 2, 2) @ (batch_size, 3, 2).transpose(-1, -2) -> (batch_size, 2, 3)
        rotated = torch.bmm(R_batch, tri_points_expanded.transpose(-1, -2))
        # Transpose back: (batch_size, 2, 3) -> (batch_size, 3, 2)
        rotated = rotated.transpose(-1, -2)
        
        # Add center position to get final triangle points
        # player_positions: (batch_size, 2) -> (batch_size, 1, 2)
        # rotated: (batch_size, 3, 2)
        final_triangle_points = rotated + player_positions.unsqueeze(1)
        
        # Combine center and triangle points: (batch_size, 4, 2)
        # player_positions: (batch_size, 2) -> (batch_size, 1, 2)
        center_expanded = player_positions.unsqueeze(1)
        agent_pos_batch = torch.cat([center_expanded, final_triangle_points], dim=1)
        
        return agent_pos_batch

    def _generate_observations_batch(self, batch_configs: dict, batch_waypoints: torch.Tensor) -> dict:
        """Generate observations for the batch with proper agent positions."""
        batch_size = batch_configs['player_positions'].shape[0]
        
        # Calculate proper agent positions (center + triangle keypoints)
        agent_positions_batch = self._get_agent_pos_batch(
            batch_configs['player_positions'], 
            batch_configs['player_orientations']
        )
        # Create observations
        observations = {
            'agent_positions': agent_positions_batch,  # (batch_size, 4, 2)
            'agent_orientations': batch_configs['player_orientations'],
            'object_positions': batch_configs['object_positions'],
            'object_types': batch_configs['object_types'],
            'waypoints': batch_waypoints,
            'done': torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        }
        
        return observations

# Usage example
def generate_efficient_data(num_batches: int = 10, batch_size: int = 32, device: str = 'cpu'):
    """Generate training data efficiently using tensorized operations."""
    
    tensorized_game = TensorizedPseudoGame(
        batch_size=batch_size,
        device=device,
        biased=True,
        augmented=True
    )
    
    all_trajectories = []
    
    for batch_idx in range(num_batches):
        print(f"Generating batch {batch_idx + 1}/{num_batches}")
        
        # Generate a full batch of trajectories
        batch_data = tensorized_game.generate_batch_trajectories()
        all_trajectories.append(batch_data)
    
    return all_trajectories

# Example usage:
trajectories = generate_efficient_data(num_batches=5, batch_size=64, device='cpu')
