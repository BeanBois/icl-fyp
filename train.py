

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
from utils import orthonormalize_2x2


# import agent
from agent_files import InstantPolicyAgent



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
            
            # # Forward pass through agent - returns normalized predictions and normalized targets
            # predicted_per_node_noise, noisy_actions = self.agent(
            #     curr_obs, context, clean_actions
            # ) 

            # Convert normalized target noise to per-node format for comparison
            # actual_pn_denoising_dir = self.actions_to_per_node_target(
            #     agent_keypoints, noisy_actions, clean_actions
            # )
            # Forward pass through agent - returns normalized predictions and normalized targets
            predicted_per_node_noise_norm, noisy_actions = self.agent(
                curr_obs, context, clean_actions
            ) 

            actual_pn_denoising_dir_norm = self.actions_to_per_node_target_normalised(
                agent_keypoints, noisy_actions, clean_actions, agent.max_flow_translation, agent.max_flow_rotation
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

    

    def actions_to_per_node_target(
        self,
        agent_keypoints,       # [4, 2], NOT necessarily centered; we'll center here
        noisy_actions,         # [T, 10] -> SE(2) (9) + state (1)
        clean_actions,         # [T, 10] -> SE(2) (9) + state (1)
        *, orthonormalize=True
    ):
        """
        Build per-node denoising targets for diffusion training.

        Returns:
            targets: [T, 4, 5] with channels:
                    [trans_x, trans_y, rot_x, rot_y, gripper_delta]
                    = (clean - noisy) components broadcasted over 4 keypoints.
        """
        device = noisy_actions.device
        dtype  = noisy_actions.dtype

        T = noisy_actions.shape[0]

        # --- unpack actions ---
        # SE(2) as 3x3 homogeneous matrices + scalar gripper state
        Rk  = noisy_actions[:, :9].view(T, 3, 3)[:, :2, :2]   # [T, 2, 2]
        tk  = noisy_actions[:, :9].view(T, 3, 3)[:, :2,  2]   # [T, 2]
        agk = noisy_actions[:, 9:10]                          # [T, 1]

        R0  = clean_actions[:, :9].view(T, 3, 3)[:, :2, :2]   # [T, 2, 2]
        t0  = clean_actions[:, :9].view(T, 3, 3)[:, :2,  2]   # [T, 2]
        ag0 = clean_actions[:, 9:10]                          # [T, 1]

        if orthonormalize:
            Rk = orthonormalize_2x2(Rk)
            R0 = orthonormalize_2x2(R0)

        # --- center keypoints (COM at origin) for rotation flow ---
        # agent_keypoints: [4,2]
        kp = agent_keypoints.to(device=device, dtype=dtype)
        kp_centered = kp 

        # --- translation delta (broadcast to all nodes) ---
        dt = (t0 - tk)                                    # [T,2]
        trans_flow = dt[:, None, :].expand(T, kp.shape[0], 2)   # [T,4,2]

        # --- rotation flow: (R0 - Rk) acting on centered keypoints ---
        # rot_flow_i = R0*kp - Rk*kp  (applied per timestep to all 4 kps)
        # Use (kp @ R^T) for row-vector convention
        rot_flow = torch.matmul(
            kp_centered[None, :, :],                      # [1,4,2]
            (R0.transpose(-2, -1) - Rk.transpose(-2, -1)) # [T,2,2]
        )                                                 # [T,4,2]

        # --- gripper delta (same for all nodes) = clean - noisy ---
        dag = (ag0 - agk)                                 # [T,1]
        g = dag[:, None, :].expand(T, kp.shape[0], 1)     # [T,4,1]

        # --- concat into [T,4,5]: [trans_x, trans_y, rot_x, rot_y, gripper_delta] ---
        targets = torch.cat([trans_flow, rot_flow, g], dim=-1)  # [T,4,5]
        return targets
    
    def actions_to_per_node_target_normalised(
        self,
        agent_keypoints,       # [4, 2], NOT necessarily centered; we'll center here
        noisy_actions,         # [T, 10] -> SE(2) (9) + state (1)
        clean_actions,
        trans_cap_m,
        theta_max_rad,
        eps = 1e-8,         # [T, 10] -> SE(2) (9) + state (1)
        *, orthonormalize=True
    ):
        """
        Build per-node denoising targets for diffusion training.

        Returns:
            targets: [T, 4, 5] with channels:
                    [trans_x, trans_y, rot_x, rot_y, gripper_delta]
                    = (clean - noisy) components broadcasted over 4 keypoints.
        """
        device = noisy_actions.device
        dtype  = noisy_actions.dtype

        T = noisy_actions.shape[0]

        # --- unpack actions ---
        # SE(2) as 3x3 homogeneous matrices + scalar gripper state
        Rk  = noisy_actions[:, :9].view(T, 3, 3)[:, :2, :2]   # [T, 2, 2]
        tk  = noisy_actions[:, :9].view(T, 3, 3)[:, :2,  2]   # [T, 2]
        agk = noisy_actions[:, 9:10]                          # [T, 1]

        R0  = clean_actions[:, :9].view(T, 3, 3)[:, :2, :2]   # [T, 2, 2]
        t0  = clean_actions[:, :9].view(T, 3, 3)[:, :2,  2]   # [T, 2]
        ag0 = clean_actions[:, 9:10]                          # [T, 1]

        if orthonormalize:
            Rk = orthonormalize_2x2(Rk)
            R0 = orthonormalize_2x2(R0)

        # --- center keypoints (COM at origin) for rotation flow ---
        # agent_keypoints: [4,2]
        kp = agent_keypoints.to(device=device, dtype=dtype)
        kp_centered = kp 
        kp_r = kp_centered.norm(dim=-1, keepdim=True).clamp_min(eps) 

        # --- translation delta (broadcast to all nodes) ---
        dt = (t0 - tk)                                    # [T,2]
        trans_flow = dt[:, None, :].expand(T, kp.shape[0], 2)   # [T,4,2]

        # --- rotation flow: (R0 - Rk) acting on centered keypoints ---
        # rot_flow_i = R0*kp - Rk*kp  (applied per timestep to all 4 kps)
        # Use (kp @ R^T) for row-vector convention
        rot_flow = torch.matmul(
            kp_centered[None, :, :],                      # [1,4,2]
            (R0.transpose(-2, -1) - Rk.transpose(-2, -1)) # [T,2,2]
        )                                                 # [T,4,2]

        # --- gripper delta (same for all nodes) = clean - noisy ---
        dag = (ag0 - agk)                                 # [T,1]
        g = dag[:, None, :].expand(T, kp.shape[0], 1)     # [T,4,1]
        # --- CAP & NORMALIZE ---
        # 1) Translation: cap & normalize by a single scalar (meters)
        trans_cap = torch.as_tensor(trans_cap_m, device=device, dtype=dtype).clamp_min(eps)
        trans_flow_n = torch.clamp(trans_flow / trans_cap, -1.0, 1.0)

        # 2) Rotation: cap in DEGREES -> radians, then per-node linear cap = theta * radius
        rot_cap_per_node = (theta_max_rad.clamp_min(eps)) * kp_r         # [4,1] in meters
        rot_cap_per_node = rot_cap_per_node[None, :, :].expand(T, -1, -1)  # [T,4,1]
        rot_flow_n = torch.clamp(rot_flow / rot_cap_per_node, -1.0, 1.0)

        # 3) Gripper: clamp to [-1,1] for safety (already small)
        g_n = torch.clamp(g, -1.0, 1.0)

        # concat: [tx,ty, rx,ry, g]
        targets = torch.cat([trans_flow_n, rot_flow_n, g_n], dim=-1)    # [T,4,5]
        return targets
    
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
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0 and save_model:
                checkpoint_path = save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        if save_model:
            torch.save(self.agent.state_dict(), save_path)
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
    print(f'training agent: {_type}, version: {version}')
    
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
        max_flow_translation=CONFIGS['MAX_DISPLACEMENT_PER_STEP'] * 2,  # 2cm = twice the 1cm max displacement
        max_flow_rotation=CONFIGS['MAX_DEGREE_PER_TURN'] * 2      # 6 degrees = twice the 3 degree max displacement
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