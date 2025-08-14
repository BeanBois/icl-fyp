import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from training_data_generator import TensorizedPseudoDemoGenerator
from training_data_generator import PseudoDemoGenerator

from agent_files import InstantPolicyAgent

class Trainer: 
    def __init__(self, agent, num_demos_for_context, num_agent_nodes=4, pred_horizon=8, 
                 min_num_waypoints=2, max_num_waypoints=6, demo_length=10, batch_size=10, device='cuda'):
        self.agent = agent
        self.device = device 
        
        self.data_generator = PseudoDemoGenerator(
            device=device,
            num_demos=num_demos_for_context + 1, 
            min_num_waypoints=min_num_waypoints,
            max_num_waypoints=max_num_waypoints,
            demo_length=demo_length,
            # batch_size=batch_size,
        )
        
        self.batch_size = batch_size
        self.pred_horizon = pred_horizon
        self.num_agent_nodes = num_agent_nodes
        
        # Optimizer (from appendix: AdamW with 1e-5 learning rate)
        self.optimizer = optim.AdamW(agent.parameters(), lr=1e-5)
        
    def train_step(self):
        """Training step following paper's SE(3) approach"""
        self.optimizer.zero_grad()
        
        # Get batch of training data
        curr_obs_batch, context_batch, clean_actions_batch = self.data_generator.get_batch_samples(self.batch_size)
        agent_keypoints = self.data_generator.get_agent_keypoints()
        
        total_loss = 0.0
        
        # Process each sample in the batch
        for i in range(self.batch_size):
            curr_obs = curr_obs_batch[i]
            context = context_batch[i] 
            clean_actions = clean_actions_batch[i]  # [T, 10] - SE(2) + state
            
            # Forward pass through agent - this handles the SE(3) noise addition internally
            # The agent's forward() method already:
            # 1. Converts SE(2) actions to se(2) tangent space (your _SE2_to_se2)
            # 2. Adds noise in tangent space 
            # 3. Converts back to SE(2) (your _se2_to_SE2)
            # 4. Predicts per-node denoising directions
            predicted_per_node_noise, action_noise = self.agent(curr_obs, context, clean_actions)
            
            # Convert action noise to per-node format for loss computation
            target_per_node_noise = self.action_noise_to_per_node_format(
                action_noise, agent_keypoints, clean_actions
            )
            
            # MSE loss between predicted and target per-node directions
            loss = nn.MSELoss()(predicted_per_node_noise, target_per_node_noise)
            scaled_loss = loss / self.batch_size
            scaled_loss.backward()
            total_loss += loss.item()
        
        self.optimizer.step()
        return total_loss / self.batch_size
    
    def action_noise_to_per_node_format(self, action_noise, agent_keypoints, clean_actions):
        """
        Convert action noise to per-node denoising directions format.
        
        This follows the paper's approach:
        1. Action noise is already in SE(2) tangent space from agent.forward()
        2. Convert this to how each gripper node should move
        3. Return in [T, 4, 5] format for loss computation
        
        Args:
            action_noise: [T, 10] - SE(2) + state noise from agent
            agent_keypoints: [4, 2] - initial gripper keypoints
            clean_actions: [T, 10] - clean actions for reference
            
        Returns:
            per_node_directions: [T, 4, 5] - per-node denoising directions
        """
        T = action_noise.shape[0]
        device = action_noise.device
        
        # Split action noise into SE(2) and state components
        se2_noise = action_noise[:, :-1].view(T, 3,3)  # [T, 9]
        state_noise = action_noise[:, -1:]  # [T, 1]
        per_node_directions = torch.zeros(T, 4, 5, device=device)
        
        for t in range(T):
            se2_matrix = se2_noise[t]  # [3, 3]
            
            # Method: Decompose SE(2) into translation and rotation components
            # This is cleaner than applying the full transformation
            
            # Extract pure translation (affects all nodes equally)
            translation = se2_matrix[:2, 2]  # [2] - [x, y]
            
            # Extract rotation matrix and compute rotation effect per node
            rotation_matrix = se2_matrix[:2, :2]  # [2, 2]
            
            # Calculate how rotation affects each keypoint
            # Since agent_keypoints are already relative to agent center, use them directly
            rotated_keypoints = agent_keypoints @ rotation_matrix.T  # [4, 2]
            rotation_displacement = rotated_keypoints - agent_keypoints  # [4, 2]
            
            # Fill per-node directions: [x_t, y_t, rx_t, ry_t, state]
            per_node_directions[t, :, 0] = translation[0]  # x translation (same for all nodes)
            per_node_directions[t, :, 1] = translation[1]  # y translation (same for all nodes)
            per_node_directions[t, :, 2] = rotation_displacement[:, 0]  # x rotation effect (different per node)
            per_node_directions[t, :, 3] = rotation_displacement[:, 1]  # y rotation effect (different per node)
            per_node_directions[t, :, 4] = state_noise[t]  # state change (same for all nodes)
        
        return per_node_directions
    
    def train_epoch(self, num_steps_per_epoch=100):
        """Train for one epoch"""
        total_loss = 0.0
        self.agent.train()
        
        for step in range(num_steps_per_epoch):
            loss = self.train_step()
            total_loss += loss
            
            if step % 10 == 0:
                print(f"Step {step}/{num_steps_per_epoch}, Loss: {loss:.6f}")
        
        return total_loss / num_steps_per_epoch
    
    def full_training(self, num_steps_per_epoch=200, num_epochs=50, 
                      save_model=True, save_path="instant_policy.pth"):
        """Full training loop"""
        avg_losses = []
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(num_steps_per_epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
            
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
            print(f"Final model saved: {save_path}")
            
        return avg_losses

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