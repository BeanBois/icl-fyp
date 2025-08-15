import torch
import torch.nn as nn 
from .instant_policy import InstantPolicy 
import numpy as np

# TODO : standardise action mode to large

# TODO : might need to fix how actions are being clamped so cooked lol
import math

def sinusoidal_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: [B] tensor of timesteps
        dim: embedding dimension
    Returns:
        [B, dim] tensor
    """
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

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

        self.t_embed_dim = self.node_embd_dim - agent_state_embd_dim
        self.t_embed_proj = nn.Sequential(
            nn.Linear(self.t_embed_dim,self.t_embed_dim, device=self.device),
            nn.SiLU(),
            nn.Linear(self.t_embed_dim, self.t_embed_dim, device=self.device)
        )

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
            nn.Sigmoid(),  # Output in [0, 1] range for binary gripper actions
        )
    
    # idk what i am doing here 
    # but in the code they kinda just make se2 actions to [xt, yt, theta] 

    def forward(self,
                curr_obs,
                context,
                clean_actions):
        B = 1  # always processing 1 sample
        timesteps = torch.randint(0, self.num_diffusion_steps, (B,), device=self.device)
        
        # get noisy action
        noisy_actions, action_noise = self._get_noisy_actions(clean_actions, timesteps)

        # Sinusoidal embedding + projection
        sin_embed = sinusoidal_embedding(timesteps, self.t_embed_dim)
        t_embed = self.t_embed_proj(sin_embed)  # [B, node_embd_dim]

        node_embs = self.policy(curr_obs, context, noisy_actions, t_embed).to(self.device)
        T, num_nodes, hidden_dim = node_embs.shape
        flat_node_embs = node_embs.view(T * num_nodes, hidden_dim)
        
        # Predict normalized noise components (outputs are already in [-1,1] due to Tanh/Sigmoid)
        flat_translation_noise_norm = self.pred_head_p(flat_node_embs).to(self.device)    # [-1, 1] 
        flat_rotation_noise_norm = self.pred_head_rot(flat_node_embs).to(self.device)     # [-1, 1] 
        flat_gripper_noise_norm = self.pred_head_g(flat_node_embs).to(self.device)       # [-1, 1]

        per_node_translation = flat_translation_noise_norm.view(T, num_nodes, 2)
        per_node_rotation = flat_rotation_noise_norm.view(T, num_nodes, 2)        
        per_node_gripper = flat_gripper_noise_norm.view(T, num_nodes, 1)


        # # Denormalize the predictions to actual noise scale
        # flat_translation_noise = self._denormalize_translation_noise(flat_translation_noise_norm)
        # flat_rotation_noise = self._denormalize_rotation_noise(flat_rotation_noise_norm)
        # flat_gripper_noise = self._denormalize_gripper_noise(flat_gripper_noise_norm)
        
        # per_node_translation = flat_translation_noise.view(T, num_nodes, 2)
        # per_node_rotation = flat_rotation_noise.view(T, num_nodes, 2)        
        # per_node_gripper = flat_gripper_noise.view(T, num_nodes, 1)

            # Combine predicted noise

        predicted_per_node_noise = torch.cat([per_node_translation, per_node_rotation, per_node_gripper], dim=-1)
    
        # we can choose to normalise both action noise and predicted_per_node_noise but for now no need
        # predicted_per_node_noise in range {self.max_flow ... } and action_noise in range {self.max_trans} 
        # we essentially limit our denoising power for a more contorlled denoising process
        return predicted_per_node_noise, noisy_actions
    
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
        return self._get_noisy_actions_large(clean_actions, timesteps)
        # else:
        # return self._get_noisy_actions_small(clean_actions, timesteps)
    ########## LARGE ACTION NOISE ADDING AUX ##########################
    
    #### NEW FIX ####
    def _get_noisy_actions_large(self, clean_actions, timesteps):
        """
        Large displacement: Add noise directly in SE(2) space with proper manifold projection
        Following the paper's guidance for "bigger displacements"
        """
        batch_size = clean_actions.shape[0]
        binary_actions = clean_actions[..., -1]
        SE2_clean_flat = clean_actions[..., :-1]
        SE2_clean = SE2_clean_flat.view(batch_size, 3, 3)
        
        # Sample noise directly as SE(2) transformations (not in tangent space!)
        # This avoids the singularity issues of SE(2) ↔ se(2) conversion
        
        # 1. Sample translation noise
        translation_noise = torch.randn(batch_size, 2, device=self.device) * self.max_flow_translation
        
        # 2. Sample rotation noise (directly as angles, avoiding tangent space)
        rotation_noise_angles = torch.randn(batch_size, device=self.device) * self.max_flow_rotation
        
        # 3. Construct SE(2) noise matrices directly
        cos_theta = torch.cos(rotation_noise_angles)
        sin_theta = torch.sin(rotation_noise_angles)
        
        SE2_noise = torch.zeros(batch_size, 3, 3, device=self.device, dtype=SE2_clean.dtype)
        SE2_noise[..., 0, 0] = cos_theta
        SE2_noise[..., 0, 1] = -sin_theta
        SE2_noise[..., 1, 0] = sin_theta
        SE2_noise[..., 1, 1] = cos_theta
        SE2_noise[..., :2, 2] = translation_noise
        SE2_noise[..., 2, 2] = 1.0
        
        # 4. Apply diffusion schedule using direct matrix operations
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1, 1)
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
        
        # 5. For large displacements: direct interpolation + manifold projection
        # This is the key difference from the small displacement method
        SE2_noisy_raw = sqrt_alpha * SE2_clean + sqrt_one_minus_alpha * SE2_noise
        
        # 6. Project back to SE(2) manifold (this is essential!)
        SE2_noisy = self._project_to_SE2_manifold(SE2_noisy_raw)
        SE2_noisy_flat = SE2_noisy.view(batch_size, 9)
        
        # Handle binary actions
        binary_noise = torch.randn_like(binary_actions)
        alpha_cumprod_binary = self.alpha_cumprod[timesteps]
        sqrt_alpha_bin = torch.sqrt(alpha_cumprod_binary)
        sqrt_one_minus_alpha_bin = torch.sqrt(1 - alpha_cumprod_binary)
        noisy_binary = sqrt_alpha_bin * binary_actions + sqrt_one_minus_alpha_bin * binary_noise
        
        # Combine results
        noisy_actions = torch.cat([SE2_noisy_flat, noisy_binary.view(-1, 1)], dim=-1)
        
        # The noise is the difference in SE(2) space (for training target)
        SE2_noise_actual = SE2_noisy_flat - SE2_clean_flat
        full_noise = torch.cat([SE2_noise_actual, binary_noise.view(-1, 1)], dim=-1)
        
        return noisy_actions, full_noise
    

    def _project_to_SE2_manifold(self, SE2_matrices):
        """
        Project matrices back to valid SE(2) manifold using SVD
        This is the crucial step for the large displacement method
        """
        batch_size = SE2_matrices.shape[0]
        device = SE2_matrices.device
        dtype = SE2_matrices.dtype
        
        # Initialize output
        projected = torch.zeros_like(SE2_matrices)
        
        # Extract the 2x2 rotation part
        R_noisy = SE2_matrices[..., :2, :2]  # [batch, 2, 2]
        
        # Project to SO(2) using SVD (closest rotation matrix)
        U, S, Vt = torch.linalg.svd(R_noisy)
        R_projected = U @ Vt  # [batch, 2, 2]
        
        # Ensure proper rotation (determinant = +1, not -1)
        det = torch.det(R_projected)  # [batch]
        
        # For determinants < 0, flip the last column of Vt
        flip_mask = det < 0
        if torch.any(flip_mask):
            Vt_corrected = Vt.clone()
            Vt_corrected[flip_mask, 1, :] *= -1  # Flip second row of Vt
            R_projected[flip_mask] = U[flip_mask] @ Vt_corrected[flip_mask]
        
        # Keep translation part unchanged
        t = SE2_matrices[..., :2, 2]  # [batch, 2]
        
        # Reconstruct valid SE(2) matrices
        projected[..., :2, :2] = R_projected
        projected[..., :2, 2] = t
        projected[..., 2, :2] = 0.0
        projected[..., 2, 2] = 1.0
        
        return projected
    

    #### OLD ####
    # def _get_noisy_actions_large(self, clean_actions, timesteps):
    #     """
    #     Large displacement: Add noise directly in SE(2) space with proper projection
    #     """
    #     binary_actions = clean_actions[..., -1]
    #     SE2_clean_flat = clean_actions[..., :-1]
    #     SE2_clean = SE2_clean_flat.view(-1,3,3)
        
    #     # Sample noise in se(2) tangent space (this is key!)
    #     batch_size = SE2_clean.shape[0]
    #     se2_noise = torch.randn(batch_size, 3, device=self.device)  # [ρx, ρy, θ]
        
        
    #     # Convert noise to SE(2) matrices using matrix exponential
    #     SE2_noise = self._se2_to_SE2(se2_noise)  # [batch, 3, 3]
        
    #     # Apply diffusion schedule
    #     alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1, 1)
        
    #     # For SE(2), we compose transformations: T_noisy = T_clean @ exp(√(1-α) * ξ)
    #     # Simplified as: T_noisy = √α * T_clean + √(1-α) * T_noise_scaled
    #     sqrt_alpha = torch.sqrt(alpha_cumprod_t)
    #     sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
        
    #     # Scale the noise transformation
    #     SE2_noise_scaled = torch.eye(3, device=self.device).unsqueeze(0) + sqrt_one_minus_alpha * (SE2_noise - torch.eye(3, device=self.device).unsqueeze(0))
        
    #     # Compose transformations (this is the proper SE(2) way)
    #     SE2_noisy = sqrt_alpha * SE2_clean + sqrt_one_minus_alpha * SE2_noise
        
    #     # Project back to SE(2) manifold (ensure rotation matrix constraints)
    #     SE2_noisy = self._project_to_SE2_manifold(SE2_noisy)
    #     SE2_noisy_flat = SE2_noisy.view(-1, 9)
        
    #     # Handle binary actions
    #     binary_noise = torch.randn_like(binary_actions)
    #     alpha_cumprod_binary = self.alpha_cumprod[timesteps].view(-1,)
    #     noisy_binary = torch.sqrt(alpha_cumprod_binary) * binary_actions + torch.sqrt(1 - alpha_cumprod_binary) * binary_noise
        
    #     # Combine results
    #     noisy_actions = torch.cat([SE2_noisy_flat, noisy_binary.view(-1,1)], dim=-1)
        
    #     # The noise is the difference in action space (for training target)
    #     moving_noise = SE2_noisy_flat - SE2_clean_flat
    #     full_noise = torch.cat([moving_noise, binary_noise.view(-1,1)], dim=-1)
        
    #     return noisy_actions, full_noise


    # def _project_to_SE2_manifold(self, SE2_matrices):
        """
        Project matrices back to valid SE(2) manifold
        """
        batch_size = SE2_matrices.shape[0]
        projected = torch.zeros_like(SE2_matrices)
        
        # Extract rotation part and re-orthogonalize using SVD
        R = SE2_matrices[..., :2, :2]  # [batch, 2, 2]
        U, S, V = torch.svd(R)
        R_proj = U @ V.transpose(-2, -1)  # Closest orthogonal matrix
        
        # Ensure proper rotation (det = 1)
        det = torch.det(R_proj)
        R_proj[det < 0] = R_proj[det < 0] @ torch.tensor([[-1, 0], [0, 1]], device=self.device, dtype=R_proj.dtype)
        
        # Keep translation as-is
        t = SE2_matrices[..., :2, 2]
        
        # Reconstruct SE(2) matrix
        projected[..., :2, :2] = R_proj
        projected[..., :2, 2] = t
        projected[..., 2, 2] = 1.0
        
        return projected
    
    ### Alternative ### 
    def _sample_SE2_noise_directly(self, batch_size):
        """
        Helper function to sample SE(2) transformations directly
        Avoids tangent space entirely for large displacements
        """
        # Sample translation components
        translation = torch.randn(batch_size, 2, device=self.device) * self.max_flow_translation
        
        # Sample rotation angles uniformly (avoiding concentration around 0)
        # For large displacements, we want uniform distribution over [-π, π]
        rotation_angles = (torch.rand(batch_size, device=self.device) - 0.5) * 2 * self.max_flow_rotation
        
        # Construct SE(2) matrices
        cos_theta = torch.cos(rotation_angles)
        sin_theta = torch.sin(rotation_angles)
        
        SE2_noise = torch.zeros(batch_size, 3, 3, device=self.device)
        SE2_noise[..., 0, 0] = cos_theta
        SE2_noise[..., 0, 1] = -sin_theta
        SE2_noise[..., 1, 0] = sin_theta
        SE2_noise[..., 1, 1] = cos_theta
        SE2_noise[..., :2, 2] = translation
        SE2_noise[..., 2, 2] = 1.0
        
        return SE2_noise

    # Alternative version with even more direct manifold operations
    def _get_noisy_actions_large_alternative(self, clean_actions, timesteps):
        """
        Alternative large displacement method using SE(2) group operations
        Even more faithful to manifold structure
        """
        batch_size = clean_actions.shape[0]
        binary_actions = clean_actions[..., -1]
        SE2_clean_flat = clean_actions[..., :-1]
        SE2_clean = SE2_clean_flat.view(batch_size, 3, 3)
        
        # Sample noise as SE(2) group elements
        SE2_noise = self._sample_SE2_noise_directly(batch_size)
        
        # Apply diffusion schedule
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1, 1)
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
        
        # Option 1: Linear interpolation + projection (simpler)
        SE2_noisy_raw = sqrt_alpha * SE2_clean + sqrt_one_minus_alpha * SE2_noise
        SE2_noisy = self._project_to_SE2_manifold(SE2_noisy_raw)
        
        # Option 2: Geodesic interpolation (more principled but complex)
        # SE2_noisy = self._geodesic_interpolation(SE2_clean, SE2_noise, sqrt_one_minus_alpha.squeeze(-1))
        
        SE2_noisy_flat = SE2_noisy.view(batch_size, 9)
        
        # Handle binary actions
        binary_noise = torch.randn_like(binary_actions)
        alpha_cumprod_binary = self.alpha_cumprod[timesteps]
        sqrt_alpha_bin = torch.sqrt(alpha_cumprod_binary)
        sqrt_one_minus_alpha_bin = torch.sqrt(1 - alpha_cumprod_binary)
        noisy_binary = sqrt_alpha_bin * binary_actions + sqrt_one_minus_alpha_bin * binary_noise
        
        # Combine results
        noisy_actions = torch.cat([SE2_noisy_flat, noisy_binary.view(-1, 1)], dim=-1)
        
        # The noise is the difference (this is what the network learns to predict)
        SE2_noise_actual = SE2_noisy_flat - SE2_clean_flat
        full_noise = torch.cat([SE2_noise_actual, binary_noise.view(-1, 1)], dim=-1)
        
        return noisy_actions, full_noise
    # unused 
    # def _get_noisy_actions_small(self, clean_actions, timesteps):
    #     """
    #     Small displacement: se(2) tangent space approach (your current approach is mostly correct)
    #     """
    #     binary_actions = clean_actions[..., -1]
    #     SE2_actions_flat = clean_actions[..., :-1]
    #     SE2_actions = clean_actions[..., :-1].view(-1, 3,3)

        
    #     # Convert to SE(2) then to se(2) tangent space
    #     se2_actions = self._SE2_to_se2(SE2_actions)
        
    #     # Normalize to [-1, 1] 
    #     se2_normalized = self._normalise_se2(se2_actions)
        
    #     # Sample noise in normalized tangent space
    #     se2_noise = torch.randn_like(se2_normalized)
    #     binary_noise = torch.randn_like(binary_actions)
        
    #     # Apply diffusion schedule
    #     alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1)
    #     sqrt_alpha = torch.sqrt(alpha_cumprod_t)
    #     sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
        
    #     # Add noise in tangent space
    #     noisy_se2_normalized = sqrt_alpha * se2_normalized + sqrt_one_minus_alpha * se2_noise

    #     sqrt_alpha_bin = sqrt_alpha.view(-1)
    #     sqrt_one_minus_alpha_bin = sqrt_one_minus_alpha.view(-1)
    #     noisy_binary = sqrt_alpha_bin * binary_actions + sqrt_one_minus_alpha_bin * binary_noise
        
    #     # Convert back: denormalize, se(2) → SE(2)
    #     noisy_se2_unnorm = self._unnormalize_se2(noisy_se2_normalized)
    #     noisy_SE2 = self._se2_to_SE2(noisy_se2_unnorm)
    #     noisy_SE2_flat = noisy_SE2.view(-1,9)
        
    #     # Combine results
    #     noisy_actions = torch.cat([noisy_SE2_flat, noisy_binary.view(-1,1)], dim=-1)
        
    #     # get noise added
    #     moving_noise = noisy_SE2_flat - SE2_actions_flat
    #     full_noise = torch.cat([moving_noise, binary_noise.view(-1,1)], dim=-1)
        
    #     return noisy_actions, full_noise
    
    # def _normalise_se2(self,se2_actions):
    #     normalized = se2_actions.clone()
    #     normalized[...,:2] /= self.max_translation
    #     normalized[..., 2:3] /= self.max_rotation
    #     return normalized

    # def _unnormalize_se2(self, normalized_se2):
    #     translation = normalized_se2[..., :2] * self.max_translation 
    #     rotation = normalized_se2[..., 2:3] * self.max_rotation
    #     return torch.cat([translation, rotation], dim=-1)