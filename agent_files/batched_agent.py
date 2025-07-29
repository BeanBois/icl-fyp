# Batched InstantPolicyAgent
import torch
import torch.nn as nn 
from .instant_policy import InstantPolicy 
import numpy as np
from typing import List, Tuple

class BatchedInstantPolicyAgent(nn.Module):
    """
    Batched version of InstantPolicyAgent that can process multiple samples simultaneously
    """
    
    def __init__(self,
                device,
                max_translation,
                num_diffusion_steps=100,
                num_agent_nodes=4, 
                pred_horizon=5, 
                num_att_heads=16,
                head_dim=64,
                agent_state_embd_dim=64,
                edge_pos_dim=2
                ):
        super(BatchedInstantPolicyAgent, self).__init__()

        self.policy = InstantPolicy(
            device=device, 
            num_agent_nodes=num_agent_nodes,
            pred_horizon=pred_horizon,
            num_att_heads=num_att_heads,
            head_dim=head_dim,
            agent_state_embd_dim=agent_state_embd_dim,
            edge_pos_dim=edge_pos_dim
        )
        self.max_translation = max_translation
        self.device = device 
        self.num_diffusion_steps = num_diffusion_steps
        self.pred_horizon = pred_horizon
        self.num_agent_nodes = num_agent_nodes
        
        # Diffusion schedule
        self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        self.alpha_schedule = 1 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)
        
        # Network dimensions
        self.hidden_dim = num_att_heads * head_dim
        self.node_embd_dim = num_att_heads * head_dim

        # Prediction heads
        self.pred_head_p = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim, 2, device=self.device),
        )
        self.pred_head_rot = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim, 1, device=self.device),
        )
        self.pred_head_g = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim, 1, device=self.device),
            nn.ReLU(),
        )

    def forward_batch(self, 
                     curr_obs_batch: List[dict],
                     context_batch: List[List],  
                     clean_actions_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward pass
        Args:
            curr_obs_batch: List of batch_size current observations
            context_batch: List of batch_size contexts (each is a list of demos)
            clean_actions_batch: [batch_size, pred_horizon, 4] clean actions
        Returns:
            predicted_noise: [batch_size, pred_horizon, 4] predicted noise
            actual_noise: [batch_size, pred_horizon, 4] actual noise added
        """
        batch_size, seq_len, action_dim = clean_actions_batch.shape
        
        # Sample timesteps for each sample in batch
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        
        # Process each sample and collect results
        predicted_noises = []
        actual_noises = []
        
        for i in range(batch_size):
            curr_obs = curr_obs_batch[i]
            context = context_batch[i] 
            clean_actions = clean_actions_batch[i]  # [pred_horizon, 4]
            timestep = timesteps[i:i+1]  # Keep batch dimension
            
            # Generate noisy actions for this sample
            noisy_actions, action_noise = self._get_noisy_actions(clean_actions.unsqueeze(0), timestep)
            
            # Forward through policy network
            node_embs = self.policy(curr_obs, context, noisy_actions.squeeze(0))  # [num_agent_nodes, hidden_dim]
            
            # Aggregate features across agent nodes
            aggregated_features = node_embs.mean(dim=0)  # [hidden_dim]
            
            # Predict noise components
            translation_noise = self.pred_head_p(aggregated_features)    # [2]
            rotation_noise = self.pred_head_rot(aggregated_features)     # [1]
            gripper_noise = self.pred_head_g(aggregated_features)        # [1]
            
            # Combine predictions [4] -> expand to [pred_horizon, 4]
            predicted_noise = torch.cat([translation_noise, rotation_noise, gripper_noise], dim=-1)
            predicted_noise = predicted_noise.unsqueeze(0).expand(seq_len, -1)  # [pred_horizon, 4]
            
            predicted_noises.append(predicted_noise)
            actual_noises.append(action_noise.squeeze(0))  # Remove batch dim
        
        # Stack all results
        predicted_noise_batch = torch.stack(predicted_noises, dim=0)  # [batch_size, pred_horizon, 4]
        actual_noise_batch = torch.stack(actual_noises, dim=0)        # [batch_size, pred_horizon, 4]
        
        return predicted_noise_batch, actual_noise_batch

    def forward(self, curr_obs, context, clean_actions):
        """
        Single sample forward pass (for backwards compatibility)
        """
        # Wrap single sample as batch
        curr_obs_batch = [curr_obs]
        context_batch = [context]
        clean_actions_batch = clean_actions.unsqueeze(0) if clean_actions.dim() == 2 else clean_actions
        
        # Process as batch
        predicted_noise, actual_noise = self.forward_batch(curr_obs_batch, context_batch, clean_actions_batch)
        
        # Return single sample results
        return predicted_noise.squeeze(0), actual_noise.squeeze(0)

    def _linear_beta_schedule(self, beta_start, beta_end, timesteps):
        """Create linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps, device=self.device)
    
    def _get_noisy_actions(self, clean_actions, timesteps, mode='large'):
        """
        Add noise to clean actions for diffusion training
        Args:
            clean_actions: [batch_size, pred_horizon, 4] or [pred_horizon, 4]
            timesteps: [batch_size] or [1]
        Returns:
            noisy_actions: same shape as clean_actions
            action_noise: noise that was added
        """
        if clean_actions.dim() == 2:
            clean_actions = clean_actions.unsqueeze(0)  # Add batch dim
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)  # Add batch dim
            
        batch_size, seq_len, action_dim = clean_actions.shape
        
        if mode == 'large':
            return self._get_noisy_actions_large(clean_actions, timesteps)
        else:
            return self._get_noisy_actions_small(clean_actions, timesteps)
    
    def _get_noisy_actions_large(self, clean_actions, timesteps):
        """Large displacement noise in SE(2) space"""
        batch_size, seq_len, action_dim = clean_actions.shape
        
        # Separate binary and motion actions
        binary_actions = clean_actions[..., 3:4]  # [batch_size, seq_len, 1]
        moving_actions = clean_actions[..., :-1]  # [batch_size, seq_len, 3]
        
        # Process each sequence in the batch
        noisy_moving_actions_list = []
        moving_noise_list = []
        
        for b in range(batch_size):
            # Convert to SE(2) matrices for this sequence
            SE2_clean = self._actions_to_SE2(moving_actions[b])  # [seq_len, 3, 3]
            
            # Sample noise in se(2) tangent space
            se2_noise = torch.randn(seq_len, 3, device=self.device)  # [seq_len, 3] -> [ρx, ρy, θ]
            
            # Scale noise for large displacements
            se2_noise[..., :2] *= 0.1  # Large translation noise (10cm)
            se2_noise[..., 2] *= 0.5   # Large rotation noise (≈30°)
            
            # Convert noise to SE(2) matrices
            SE2_noise = self._se2_to_SE2(se2_noise)  # [seq_len, 3, 3]
            
            # Apply diffusion schedule
            alpha_cumprod_t = self.alpha_cumprod[timesteps[b]].view(1, 1, 1)  # [1, 1, 1]
            sqrt_alpha = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
            
            # Scale the noise transformation
            eye_batch = torch.eye(3, device=self.device).unsqueeze(0).expand(seq_len, -1, -1)
            SE2_noise_scaled = eye_batch + sqrt_one_minus_alpha * (SE2_noise - eye_batch)

            # Compose transformations
            SE2_noisy = sqrt_alpha * SE2_clean + sqrt_one_minus_alpha * SE2_noise
            
            # Project back to SE(2) manifold
            SE2_noisy = self._project_to_SE2_manifold(SE2_noisy)
            # Convert back to actions
            noisy_moving = self._SE2_to_actions(SE2_noisy)  # [seq_len, 3]
            moving_noise = noisy_moving - moving_actions[b]  # [seq_len, 3]
            
            noisy_moving_actions_list.append(noisy_moving)
            moving_noise_list.append(moving_noise)
        
        # Stack batch results
        noisy_moving_actions = torch.stack(noisy_moving_actions_list, dim=0)  # [batch_size, seq_len, 3]
        moving_noise = torch.stack(moving_noise_list, dim=0)  # [batch_size, seq_len, 3]
        
        # Handle binary actions
        binary_noise = torch.randn_like(binary_actions)  # [batch_size, seq_len, 1]
        alpha_cumprod_binary = self.alpha_cumprod[timesteps].view(batch_size, 1, 1)  # [batch_size, 1, 1]
        noisy_binary = (torch.sqrt(alpha_cumprod_binary) * binary_actions + 
                       torch.sqrt(1 - alpha_cumprod_binary) * binary_noise)
        
        # Combine results
        noisy_actions = torch.cat([noisy_moving_actions, noisy_binary], dim=-1)  # [batch_size, seq_len, 4]
        full_noise = torch.cat([moving_noise, binary_noise], dim=-1)  # [batch_size, seq_len, 4]
        
        return noisy_actions, full_noise

    def _get_noisy_actions_small(self, clean_actions, timestepts):
        """Small displacement noise in se(2) tangent space - simplified version"""
        # For now, implement a simplified version that works in action space directly
        batch_size, seq_len, action_dim = clean_actions.shape
        
        # Sample noise directly in action space
        action_noise = torch.randn_like(clean_actions)
        
        # Apply diffusion schedule
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
        
        # Add noise
        noisy_actions = sqrt_alpha * clean_actions + sqrt_one_minus_alpha * action_noise
        
        return noisy_actions, action_noise

    # SE(2) helper functions (vectorized for batch processing)
    def _actions_to_SE2(self, actions):
        """Convert actions [N, 3] to SE(2) matrices [N, 3, 3]"""
        num_actions = actions.shape[0]
        se2_actions = torch.zeros(num_actions, 3, 3, device=self.device)
        
        angles = actions[:, 2]
        x_trans = actions[:, 0]
        y_trans = actions[:, 1]

        se2_actions[:, 0, 2] = x_trans
        se2_actions[:, 1, 2] = y_trans
        se2_actions[:, 2, 2] = 1

        _sin_theta = torch.sin(angles)
        _cos_theta = torch.cos(angles)
        se2_actions[:, 0, 0] = _cos_theta
        se2_actions[:, 0, 1] = -_sin_theta
        se2_actions[:, 1, 0] = _sin_theta
        se2_actions[:, 1, 1] = _cos_theta

        return se2_actions

    def _SE2_to_actions(self, se2_matrices):
        """Convert SE(2) matrices [N, 3, 3] to actions [N, 3]"""
        num_actions = se2_matrices.shape[0]
        actions = torch.zeros(num_actions, 3, device=self.device)
        
        # Extract translation components
        actions[:, 0] = se2_matrices[:, 0, 2]  # x_trans
        actions[:, 1] = se2_matrices[:, 1, 2]  # y_trans
        
        # Extract rotation angle
        angles_rad = torch.atan2(se2_matrices[:, 1, 0], se2_matrices[:, 0, 0])
        actions[:, 2] = angles_rad
        
        return actions

    def _se2_to_SE2(self, xi):
        """Convert se(2) tangent vectors [N, 3] to SE(2) matrices [N, 3, 3]"""
        batch_size = xi.shape[0]
        
        rho = xi[..., :2]  # [N, 2] - [rho_x, rho_y]
        theta = xi[..., 2]  # [N] - rotation angles
        
        # Compute rotation matrices
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        R = torch.zeros(batch_size, 2, 2, device=self.device)
        R[:, 0, 0] = cos_theta
        R[:, 0, 1] = -sin_theta
        R[:, 1, 0] = sin_theta
        R[:, 1, 1] = cos_theta
        
        # Compute V matrix for translation (simplified for small angles)
        eps = 1e-6
        small_angle_mask = torch.abs(theta) < eps
        
        V = torch.zeros(batch_size, 2, 2, device=self.device)
        
        # For non-small angles
        not_small = ~small_angle_mask
        if torch.any(not_small):
            theta_nz = theta[not_small]
            sin_nz = sin_theta[not_small]
            cos_nz = cos_theta[not_small]
            
            V[not_small, 0, 0] = sin_nz / theta_nz
            V[not_small, 0, 1] = -(1 - cos_nz) / theta_nz
            V[not_small, 1, 0] = (1 - cos_nz) / theta_nz
            V[not_small, 1, 1] = sin_nz / theta_nz
        
        # For small angles, V ≈ I + (1/2) * [[0, -theta], [theta, 0]]
        if torch.any(small_angle_mask):
            V[small_angle_mask, 0, 0] = 1.0
            V[small_angle_mask, 1, 1] = 1.0
            V[small_angle_mask, 0, 1] = -0.5 * theta[small_angle_mask]
            V[small_angle_mask, 1, 0] = 0.5 * theta[small_angle_mask]
        
        # Compute translation: t = V @ rho
        t = torch.einsum('bij,bj->bi', V, rho)
        
        # Construct SE(2) matrices
        T = torch.zeros(batch_size, 3, 3, device=self.device)
        T[:, :2, :2] = R
        T[:, :2, 2] = t
        T[:, 2, 2] = 1.0
        
        return T

    def _project_to_SE2_manifold(self, SE2_matrices):
        """Project matrices back to valid SE(2) manifold using SVD"""
        batch_size = SE2_matrices.shape[0]
        projected = torch.zeros_like(SE2_matrices)
        
        # Extract rotation part and re-orthogonalize using SVD
        R = SE2_matrices[..., :2, :2]  # [batch_size, 2, 2]
        
        try:
            U, S, V = torch.svd(R)
            R_proj = U @ V.transpose(-2, -1)  # Closest orthogonal matrix
            
            # Ensure proper rotation (det = 1)
            det = torch.det(R_proj)
            correction = torch.tensor([[-1, 0], [0, 1]], device=self.device, dtype=R_proj.dtype)
            R_proj[det < 0] = R_proj[det < 0] @ correction
            
        except:
            # Fallback: use original rotation if SVD fails
            R_proj = R
        
        # Keep translation as-is
        t = SE2_matrices[..., :2, 2]
        
        # Reconstruct SE(2) matrix
        projected[..., :2, :2] = R_proj
        projected[..., :2, 2] = t
        projected[..., 2, 2] = 1.0
        
        return projected