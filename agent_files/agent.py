import torch
import torch.nn as nn 
from .instant_policy import InstantPolicy 
import numpy as np

# TODO : standardise action mode to large

# TODO : might need to fix how actions are being clamped so cooked lol
class InstantPolicyAgent(nn.Module):

    def __init__(self,
                device,
                max_translation,
                max_rotation,
                geometry_encoder,
                # New parameters for output normalization
                max_flow_translation = 100,  # 100px = twice the 10px max displacement PSEUDO_MAX_TRANLSATION
                max_flow_rotation = 36,     # 36 deg = twice the 5 degree max displacement PSEUDO_MAX_ROTATION
                num_diffusion_steps = 100,
                num_agent_nodes = 4, 
                pred_horizon = 5, 
                num_att_heads = 16,
                head_dim = 64,
                agent_state_embd_dim = 64,
                edge_pos_dim = 2,
                ):
        super(InstantPolicyAgent, self).__init__()

        
        self.policy = InstantPolicy(
                geometry_encoder=geometry_encoder,
                device=device, 
                num_agent_nodes=num_agent_nodes,
                pred_horizon=pred_horizon,
                num_att_heads=num_att_heads,
                head_dim=head_dim,
                agent_state_embd_dim=agent_state_embd_dim,
                edge_pos_dim=edge_pos_dim
                )
        
        # unused params for now 
        self.max_translation = max_translation
        self.max_rotation = torch.deg2rad(torch.tensor([max_rotation], device=device))

        self.device = device 
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        self.alpha_schedule = 1-self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)
        self.hidden_dim = num_att_heads * head_dim
        self.node_embd_dim = num_att_heads * head_dim

        # Output normalization parameters
        self.max_flow_translation = max_flow_translation  # cm
        self.max_flow_rotation = torch.deg2rad(torch.tensor([max_flow_rotation], device=device))  # radians


        self.pred_head_p = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,2, device=self.device),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        self.pred_head_rot = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim, 2, device=self.device),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        self.pred_head_g = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim,device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,1,device=self.device),
            nn.Tanh(),  # Output in [-1, 1] range for binary gripper actions
        )
    
    # idk what i am doing here 
    # but in the code they kinda just make se2 actions to [xt, yt, theta] 

    def forward(self,
                   curr_obs,
                   context,
                   clean_actions):
        batch_size = len(clean_actions)
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        
        # get noisy action
        noisy_actions, action_noise = self._get_noisy_actions(clean_actions, timesteps)

        node_embs = self.policy(curr_obs, context, noisy_actions).to(self.device)
        T, num_nodes, hidden_dim = node_embs.shape
        flat_node_embs = node_embs.view(T * num_nodes, hidden_dim)

        # Predict normalized noise components (outputs are already in [-1,1] due to Tanh/Sigmoid)
        flat_translation_noise_norm = self.pred_head_p(flat_node_embs).to(self.device)    # [-1, 1] 
        flat_rotation_noise_norm = self.pred_head_rot(flat_node_embs).to(self.device)     # [-1, 1] 
        flat_gripper_noise_norm = self.pred_head_g(flat_node_embs).to(self.device)       # [-1, 1]

        # Denormalize the predictions to actual noise scale
        flat_translation_noise = self._denormalize_translation_noise(flat_translation_noise_norm)
        flat_rotation_noise = self._denormalize_rotation_noise(flat_rotation_noise_norm)
        flat_gripper_noise = self._denormalize_gripper_noise(flat_gripper_noise_norm)
        
        per_node_translation = flat_translation_noise.view(T, num_nodes, 2)
        per_node_rotation = flat_rotation_noise.view(T, num_nodes, 2)        
        per_node_gripper = flat_gripper_noise.view(T, num_nodes, 1)

        # Combine predicted noise
        predicted_per_node_noise = torch.cat([per_node_translation, per_node_rotation, per_node_gripper], dim=-1)
    
        # we can choose to normalise both action noise and predicted_per_node_noise but for now no need
        # predicted_per_node_noise in range {self.max_flow ... } and action_noise in range {self.max_trans} 
        # we essentially limit our denoising power for a more contorlled denoising process
        return predicted_per_node_noise, action_noise
    
    def predict(self,
            curr_obs,
            context,
            noisy_actions):
        batch_size = len(noisy_actions)
        node_embs = self.policy(curr_obs, context, noisy_actions).to(self.device)
        T, num_nodes, hidden_dim = node_embs.shape
        flat_node_embs = node_embs.view(T * num_nodes, hidden_dim)

        # Predict normalized noise components (outputs are already in [-1,1] due to Tanh/Sigmoid)
        flat_translation_noise_norm = self.pred_head_p(flat_node_embs).to(self.device)    # [-1, 1] 
        flat_rotation_noise_norm = self.pred_head_rot(flat_node_embs).to(self.device)     # [-1, 1] 
        flat_gripper_noise_norm = self.pred_head_g(flat_node_embs).to(self.device)       # [-1, 1]

        # Denormalize the predictions to actual noise scale
        flat_translation_noise = self._denormalize_translation_noise(flat_translation_noise_norm)
        flat_rotation_noise = self._denormalize_rotation_noise(flat_rotation_noise_norm)
        flat_gripper_noise = self._denormalize_gripper_noise(flat_gripper_noise_norm)
        
        per_node_translation = flat_translation_noise.view(T, num_nodes, 2)
        per_node_rotation = flat_rotation_noise.view(T, num_nodes, 2)        
        per_node_gripper = flat_gripper_noise.view(T, num_nodes, 1)

        # Combine predicted noise
        predicted_per_node_noise = torch.cat([per_node_translation, per_node_rotation, per_node_gripper], dim=-1)
    
        # we can choose to normalise both action noise and predicted_per_node_noise but for now no need
        # predicted_per_node_noise in range {self.max_flow ... } and action_noise in range {self.max_trans} 
        # we essentially limit our denoising power for a more contorlled denoising process
        return predicted_per_node_noise

    def _denormalize_translation_noise(self, normalized_noise):
        """Convert normalized translation noise [-1,1] to actual scale"""
        return normalized_noise * self.max_flow_translation

    def _denormalize_rotation_noise(self, normalized_noise):
        """Convert normalized rotation noise [-1,1] to actual scale"""  
        return normalized_noise * self.max_flow_rotation

    def _denormalize_gripper_noise(self, normalized_noise):
        """Convert normalized gripper noise [0,1] to actual scale"""
        # For binary gripper, we can keep it in [0,1] or scale as needed
        return normalized_noise * 2.0 - 1.0  # Convert [0,1] to [-1,1] if needed

    def _linear_beta_schedule(self, beta_start, beta_end, timesteps):
        """Create linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps, device = self.device)

    def _SE2_to_se2(self,T, eps=1e-6):
        # Handle batch dimensions
        batch_shape = T.shape[:-2]
        
        # Extract rotation matrix (top-left 2x2)
        R = T[..., :2, :2]
        
        # Extract translation vector (top-right 2x1)
        t = T[..., :2, 2]
        
        # Compute rotation angle using atan2
        theta = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        
        # Compute V_inverse matrix for translation
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # Handle small angle case for numerical stability
        small_angle_mask = torch.abs(theta) < eps
        
        # V_inverse for general case
        V_inv = torch.zeros(*batch_shape, 2, 2, device=T.device, dtype=T.dtype)
        
        # For non-small angles
        not_small = ~small_angle_mask
        if torch.any(not_small):
            theta_nz = theta[not_small]
            sin_nz = sin_theta[not_small]
            cos_nz = cos_theta[not_small]
            
            # V_inv = (1/theta) * [[sin(theta), cos(theta)-1], [1-cos(theta), sin(theta)]]
            V_inv[not_small, 0, 0] = sin_nz / theta_nz
            V_inv[not_small, 0, 1] = (cos_nz - 1) / theta_nz
            V_inv[not_small, 1, 0] = (1 - cos_nz) / theta_nz
            V_inv[not_small, 1, 1] = sin_nz / theta_nz
        
        # For small angles, V_inv ≈ I - (1/2) * [[0, -theta], [theta, 0]]
        if torch.any(small_angle_mask):
            V_inv[small_angle_mask, 0, 0] = 1.0
            V_inv[small_angle_mask, 1, 1] = 1.0
            V_inv[small_angle_mask, 0, 1] = -0.5 * theta[small_angle_mask]
            V_inv[small_angle_mask, 1, 0] = 0.5 * theta[small_angle_mask]
        
        # Compute rho = V_inv @ t
        rho = torch.einsum('...ij,...j->...i', V_inv, t)
        
        # Stack to form se(2) vector [rho_x, rho_y, theta]
        xi = torch.stack([rho[..., 0], rho[..., 1], theta], dim=-1)
        
        return xi

    def _se2_to_SE2(self,xi):
        """
        Convert se(2) tangent vector to SE(2) matrix using exponential map.
        
        Args:
            xi: se(2) tangent vector of shape (..., 3) as [rho_x, rho_y, theta]
        
        Returns:
            T: SE(2) transformation matrix of shape (..., 3, 3)
        """
        batch_shape = xi.shape[:-1]
        
        rho = xi[..., :2]  # [rho_x, rho_y]
        theta = xi[..., 2]  # theta
        
        # Compute rotation matrix
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        R = torch.zeros(*batch_shape, 2, 2, device=xi.device, dtype=xi.dtype)
        R[..., 0, 0] = cos_theta
        R[..., 0, 1] = -sin_theta
        R[..., 1, 0] = sin_theta
        R[..., 1, 1] = cos_theta
        
        # Compute V matrix for translation
        eps = 1e-6
        small_angle_mask = torch.abs(theta) < eps
        
        V = torch.zeros(*batch_shape, 2, 2, device=xi.device, dtype=xi.dtype)
        
        # For non-small angles
        not_small = ~small_angle_mask
        if torch.any(not_small):
            theta_nz = theta[not_small]
            sin_nz = sin_theta[not_small]
            cos_nz = cos_theta[not_small]
            
            # V = (1/theta) * [[sin(theta), -(1-cos(theta))], [1-cos(theta), sin(theta)]]
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
        t = torch.einsum('...ij,...j->...i', V, rho)
        
        # Construct SE(2) matrix
        T = torch.zeros(*batch_shape, 3, 3, device=xi.device, dtype=xi.dtype)
        T[..., :2, :2] = R
        T[..., :2, 2] = t
        T[..., 2, 2] = 1.0
        
        return T

    def _get_noisy_actions(self, clean_actions, timesteps):
        """
        Choose mode based on displacement magnitude
        """
        # if mode == 'large':
            # return self._get_noisy_actions_large(clean_actions, timesteps)
        # else:
        return self._get_noisy_actions_small(clean_actions, timesteps)
        
    def _get_noisy_actions_small(self, clean_actions, timesteps):
        """
        Small displacement: se(2) tangent space approach (your current approach is mostly correct)
        """
        binary_actions = clean_actions[..., -1]
        SE2_actions_flat = clean_actions[..., :-1]
        SE2_actions = clean_actions[..., :-1].view(-1, 3,3)

        
        # Convert to SE(2) then to se(2) tangent space
        se2_actions = self._SE2_to_se2(SE2_actions)
        
        # Normalize to [-1, 1] 
        se2_normalized = self._normalise_se2(se2_actions)
        
        # Sample noise in normalized tangent space
        se2_noise = torch.randn_like(se2_normalized)
        binary_noise = torch.randn_like(binary_actions)
        
        # Apply diffusion schedule
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1)
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
        
        # Add noise in tangent space
        noisy_se2_normalized = sqrt_alpha * se2_normalized + sqrt_one_minus_alpha * se2_noise

        sqrt_alpha_bin = sqrt_alpha.view(-1)
        sqrt_one_minus_alpha_bin = sqrt_one_minus_alpha.view(-1)
        noisy_binary = sqrt_alpha_bin * binary_actions + sqrt_one_minus_alpha_bin * binary_noise
        
        # Convert back: denormalize, se(2) → SE(2)
        noisy_se2_unnorm = self._unnormalize_se2(noisy_se2_normalized)
        noisy_SE2 = self._se2_to_SE2(noisy_se2_unnorm)
        noisy_SE2_flat = noisy_SE2.view(-1,9)
        
        # Combine results
        noisy_actions = torch.cat([noisy_SE2_flat, noisy_binary.view(-1,1)], dim=-1)
        
        # get noise added
        moving_noise = noisy_SE2_flat - SE2_actions_flat
        full_noise = torch.cat([moving_noise, binary_noise.view(-1,1)], dim=-1)
        
        return noisy_actions, full_noise
    
    def _normalise_se2(self,se2_actions):
        normalized = se2_actions.clone()
        normalized[...,:2] /= self.max_flow_translation
        normalized[..., 2:3] /= self.max_rotation
        return normalized

    def _unnormalize_se2(self, normalized_se2):
        translation = normalized_se2[..., :2] * self.max_flow_translation 
        rotation = normalized_se2[..., 2:3] * self.max_rotation
        return torch.cat([translation, rotation], dim=-1)