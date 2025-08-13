# import game
from configs import PSEUDO_SCREEN_HEIGHT, PSEUDO_SCREEN_WIDTH, PseudoGame, action_mode

# import agent
from agent_files import InstantPolicyAgent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple, Dict
from training_data_generator import TensorizedPseudoDemoGenerator
# move these constants to CONFIG FILES 

# import game
from configs import PSEUDO_SCREEN_HEIGHT, PSEUDO_SCREEN_WIDTH, PseudoGame, action_mode

# import agent
from agent_files import InstantPolicyAgentNormAction as InstantPolicyAgent


class Trainer: 

    def __init__(self, agent, num_demos_for_context, num_agent_nodes = 4, pred_horizon = 8, 
                 min_num_waypoints = 2, max_num_waypoints = 6, demo_length=10, batch_size = 10, device='cuda'):
        self.agent = agent
        self.device = device 
        
        self.data_generator = TensorizedPseudoDemoGenerator(
            device=device,
            num_demos= num_demos_for_context + 1, 
            min_num_waypoints= min_num_waypoints,
            max_num_waypoints= max_num_waypoints,
            demo_length= demo_length,
            batch_size=batch_size,
        )
        
        self.agent_keypoint = None
        self.batch_size = batch_size
        self.pred_horizon = pred_horizon
        self.num_agent_nodes = num_agent_nodes
        
        # Optimizer (from appendix: AdamW with 1e-5 learning rate)
        self.optimizer = optim.AdamW(agent.parameters(), lr=1e-5)
        
        # Get normalization parameters from agent for consistent processing
        self.max_flow_translation = agent.max_flow_translation
        self.max_flow_rotation = agent.max_flow_rotation
        
    def train_step(self):
        self.optimizer.zero_grad()
        total_loss = 0.0
        
        # Get batch of training data
        curr_obs_batch, context_batch, clean_actions_batch = self.data_generator.get_batch_samples(self.batch_size)
        agent_keypoints = self.data_generator.get_agent_keypoints()
        
        # Process each sample in the batch
        for i in range(self.batch_size):
            curr_obs, context, clean_actions = curr_obs_batch[i], context_batch[i], clean_actions_batch[i]
            
            # Forward pass through agent - returns normalized predictions and normalized targets
            predicted_per_node_noise_norm, action_noise_normalized = self.agent(
                curr_obs, context, clean_actions, action_mode
            ) 
            
            # Convert normalized target noise to per-node format for comparison
            actual_pn_denoising_dir_norm = self.get_normalized_pnn_from_noise(
                agent_keypoints, action_noise_normalized
            )
            
            assert predicted_per_node_noise_norm.shape == actual_pn_denoising_dir_norm.shape
            
            # MSE loss between normalized predictions and normalized targets
            loss = nn.MSELoss()(predicted_per_node_noise_norm, actual_pn_denoising_dir_norm)
            scaled_loss = loss / self.batch_size
            scaled_loss.backward()
            total_loss += loss.item()
        
        # Single optimizer step after accumulating gradients from all batch samples
        self.optimizer.step()
        return total_loss / self.batch_size
    
    def get_normalized_pnn_from_noise(self, agent_keypoints, action_noise_normalized):
        """
        Calculate per-node denoising directions from normalized action noise
        
        Args:
            agent_keypoints: (4, 2) - current gripper keypoints [x, y]
            action_noise_normalized: (L, 10) - normalized noisy actions
        
        Returns:
            denoising_directions_normalized: (L, 4, 5) - normalized per node denoising directions
        """
        L = action_noise_normalized.shape[0]
        
        # First denormalize the action noise to get actual physical units
        action_noise_physical = self.denormalize_action_noise(action_noise_normalized)
        
        # Calculate per-node noise in physical units
        denoising_directions_physical = self.get_pnn_from_noise_physical(
            agent_keypoints, action_noise_physical
        )
        
        # Normalize the per-node directions for loss computation
        denoising_directions_normalized = self.normalize_per_node_directions(
            denoising_directions_physical
        )
        
        return denoising_directions_normalized
    
    def denormalize_action_noise(self, action_noise_normalized):
        """Convert normalized action noise back to physical units"""
        # Split the normalized noise
        se2_noise_norm = action_noise_normalized[:, :-1]  # First 9 components
        gripper_noise_norm = action_noise_normalized[:, -1:]  # Last component
        
        # Denormalize SE(2) components (this is simplified - you may need more sophisticated approach)
        # For SE(2) matrices, we need to carefully denormalize each component
        se2_noise_physical = se2_noise_norm * self.max_flow_translation  # Simplified scaling
        
        # Denormalize gripper (convert from [0,1] back to [-1,1] range)
        gripper_noise_physical = gripper_noise_norm * 2.0 - 1.0
        
        return torch.cat([se2_noise_physical, gripper_noise_physical], dim=-1)
    
    def normalize_per_node_directions(self, denoising_directions_physical):
        """Normalize per-node denoising directions to [-1,1] range"""
        L, num_nodes, _ = denoising_directions_physical.shape
        
        # Split components
        translation_shifts = denoising_directions_physical[:, :, :2]  # (L, 4, 2)
        rotation_shifts = denoising_directions_physical[:, :, 2:4]    # (L, 4, 2)
        gripper_shifts = denoising_directions_physical[:, :, 4:]      # (L, 4, 1)
        
        # Normalize each component independently
        translation_norm = torch.clamp(translation_shifts / self.max_flow_translation, -1.0, 1.0)
        rotation_norm = torch.clamp(rotation_shifts / self.max_flow_rotation, -1.0, 1.0)
        gripper_norm = torch.clamp((gripper_shifts + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] to [0,1]
        
        return torch.cat([translation_norm, rotation_norm, gripper_norm], dim=-1)
    
    def get_pnn_from_noise_physical(self, agent_keypoints, action_noise):
        """
        Calculate per-node denoising directions from noisy actions in physical units
        
        Args:
            agent_keypoints: (4, 2) - current gripper keypoints [x, y]
            action_noise: (L, 10) - noisy cumulative actions [SE(2) + state] in physical units
        
        Returns:
            denoising_directions: (L, 4, 5) - per node denoising directions in physical units
        """
        L = action_noise.shape[0]
        
        # Original keypoints repeated for each action
        original_keypoints = agent_keypoints.unsqueeze(0).repeat(L, 1, 1)  # (L, 4, 2)
        
        # Extract SE(2) transformations (first 9 dims)
        se2_noise = action_noise[:, :9].view(L, 3, 3)  # (L, 3, 3)
        state_noise = action_noise[:, 9:10]  # (L, 1)
        
        # Convert to homogeneous coordinates
        ones = torch.ones(L, 4, 1, device=agent_keypoints.device, dtype=agent_keypoints.dtype)
        keypoints_homo = torch.cat([original_keypoints, ones], dim=-1)  # (L, 4, 3)
        
        # Apply SE(2) transformations
        transformed_keypoints_homo = torch.bmm(keypoints_homo, se2_noise.transpose(-2, -1))
        transformed_keypoints = transformed_keypoints_homo[:, :, :2]  # (L, 4, 2)
        
        # Calculate translation shifts (actual displacement)
        translation_shifts = transformed_keypoints - original_keypoints  # (L, 4, 2)
        
        # Calculate rotation components (for denoising)
        # Extract rotation part of SE(2) matrix
        rotation_matrices = se2_noise[:, :2, :2]  # (L, 2, 2)
        
        # Calculate rotation-induced shifts for each keypoint
        centered_keypoints = original_keypoints - original_keypoints.mean(dim=1, keepdim=True)
        rotated_centered = torch.bmm(centered_keypoints, rotation_matrices.transpose(-2, -1))
        rotation_shifts = rotated_centered - centered_keypoints  # (L, 4, 2)
        
        # Repeat gripper state for each keypoint
        gripper_shifts = state_noise.unsqueeze(1).repeat(1, 4, 1)  # (L, 4, 1)
        
        # Combine all denoising directions: [trans_x, trans_y, rot_x, rot_y, gripper]
        denoising_directions = torch.cat([
            translation_shifts,      # (L, 4, 2) - [delta_x, delta_y]
            rotation_shifts,         # (L, 4, 2) - [delta_rot_x, delta_rot_y] 
            gripper_shifts          # (L, 4, 1) - [delta_gripper]
        ], dim=-1)  # (L, 4, 5)
        
        return denoising_directions
    
    def train_epoch(self, num_steps_per_epoch = 100):
        total_loss = 0.0
        for step in range(num_steps_per_epoch):
            # Train step
            loss = self.train_step()
            total_loss += loss
            
            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss:.6f}")
        
        return total_loss / num_steps_per_epoch
    
    def full_training(self, num_steps_per_epoch=200, num_epochs=50, 
                      save_model = True, save_path = "instant_policy.pth"):
        avg_losses = []
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(num_steps_per_epoch)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
            
            # Learning rate cooldown in final epochs
            if epoch > num_epochs - 4:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.99
                    
            avg_losses.append(avg_loss)
            
        if save_model:
            torch.save(self.agent, save_path)
        self.plot_losses(avg_losses, num_steps_per_epoch)

    def plot_losses(self, losses, num_steps_per_epoch, display = False):
        fig = plt.figure()
        plt.plot(losses)
        plt.title(f'Average losses from each epoch ({num_steps_per_epoch} steps)')
        if display:
            plt.show()
        plt.savefig(CONFIGS['FIG_FILENAME'])


# Usage example:
if __name__ == "__main__":
    from configs import CONFIGS, version, _type, geo_version
    from agent_files import GeometryEncoder2D, full_train, initialise_geometry_encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # print statement
    print(f'training agent: {_type}, version: {version}, action mode : {action_mode}')
    
    # train geometry encoder
    geometry_encoder_filename = f'geometry_encoder_2d_v{geo_version}.pth'
    node_embd_dim = CONFIGS['NUM_ATT_HEADS'] * CONFIGS['HEAD_DIM']
    grouping_radius = CONFIGS['GROUPING_RADIUS']
    num_sampled_pc = CONFIGS['NUM_SAMPLED_POINTCLOUDS']
    
    if CONFIGS['TRAIN_GEO_ENCODER']:
        full_train(num_sampled_pc, node_embd_dim, device, grouping_radius, 
                  filename=geometry_encoder_filename, num_epochs= CONFIGS['GEO_NUM_EPOCHS'], 
                  num_samples= CONFIGS['GEO_BATCH_SIZE'])

    model = GeometryEncoder2D(num_centers=num_sampled_pc, radius=grouping_radius, 
                             node_embd_dim=node_embd_dim, device=device).to(device)
    geometry_encoder = initialise_geometry_encoder(model, geometry_encoder_filename, device=device)

    # Initialize agent with normalization parameters
    agent = InstantPolicyAgent(
        device=device, 
        geometry_encoder=geometry_encoder,
        max_translation=CONFIGS['TRAINING_MAX_TRANSLATION'],
        max_rotation=CONFIGS['MAX_ROTATION_DEG'],
        num_diffusion_steps=CONFIGS['NUM_DIFFUSION_STEPS'],
        num_agent_nodes=CONFIGS['NUM_AGENT_NODES'],
        pred_horizon=CONFIGS['PRED_HORIZON'],
        num_att_heads=CONFIGS['NUM_ATT_HEADS'],
        head_dim=CONFIGS['HEAD_DIM'],
        agent_state_embd_dim=CONFIGS['AGENT_STATE_EMB_DIM'],
        edge_pos_dim=CONFIGS['EDGE_POS_DIM'],

        # Add normalization parameters
        max_flow_translation=CONFIGS['PIXEL_PER_STEP'] * 2,  # 2cm = twice the 1cm max displacement
        max_flow_rotation=CONFIGS['DEGREE_PER_TURN'] * 2      # 6 degrees = twice the 3 degree max displacement
    )
    
    # Initialize trainer
    trainer = Trainer(agent,
                    device=device, 
                    num_agent_nodes=CONFIGS['NUM_AGENT_NODES'],
                    pred_horizon=CONFIGS['PRED_HORIZON'],
                    num_demos_for_context=CONFIGS['NUM_DEMO_GIVEN'],
                    max_num_waypoints=CONFIGS['MAX_NUM_WAYPOINTS'],
                    min_num_waypoints=CONFIGS['MIN_NUM_WAYPOINTS'],
                    demo_length=CONFIGS['DEMO_MAX_LENGTH'],
                    batch_size=CONFIGS['BATCH_SIZE']
                    )
    
    # Train the model
    trainer.full_training(
        save_path=CONFIGS['MODEL_FILE_PATH'],
        save_model=CONFIGS['SAVE_MODEL'],
        num_steps_per_epoch=CONFIGS['NUM_STEPS_PER_EPOCH'],
        num_epochs=CONFIGS['NUM_EPOCHS']
    )