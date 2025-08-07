
import torch
import torch.nn as nn 
from .instant_policy import InstantPolicy 

# main agent file
# the noise addition process is done in SE2 space
# 
class InstantPolicyAgent(nn.Module):

    def __init__(self,
                device,
                max_translation,
                geometry_encoder,
                num_diffusion_steps = 100,
                num_agent_nodes = 4, 
                pred_horizon = 5, 
                num_att_heads = 16,
                head_dim = 64,
                agent_state_embd_dim = 64,
                edge_pos_dim = 2
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
        self.max_translation = max_translation
        self.device = device 
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        self.alpha_schedule = 1-self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)
        self.hidden_dim = num_att_heads * head_dim
        self.node_embd_dim = num_att_heads * head_dim

        self.pred_head_p = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,2, device=self.device),
        )
        self.pred_head_rot = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim, 2, device=self.device),
        )
        self.pred_head_g = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim,device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,1,device=self.device),
            nn.ReLU(),
        )
    
    # right now action is x,y,theta,state_change
    # optimally want to produce:  
    # x_cetner_mass_trans, y_centre_mass_trans, 
    def forward(self,
                curr_obs,
                context,
                clean_actions,
                prev_stacked_actions):
        # need to use clean actions to generate noisy actions 
        batch_size = len(clean_actions)
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        noisy_actions, action_noise = self._get_noisy_actions(clean_actions, timesteps, mode='large')

        node_embs = self.policy(curr_obs, context, noisy_actions, prev_stacked_actions).to(self.device) # N x self.num_agent_nodes x self.node_emb_dim
        N, num_nodes, hidden_dim = node_embs.shape
        flat_node_embs = node_embs.view(N * num_nodes, hidden_dim)  # [N*4, 1024]

        # Predict noise components
        flat_translation_noise = self.pred_head_p(flat_node_embs).to(self.device)    # [N, 2]
        flat_rotation_noise = self.pred_head_rot(flat_node_embs).to(self.device)    # [N, 1] 
        flat_gripper_noise = self.pred_head_g(flat_node_embs).to(self.device)    # [N, 1]
        

        per_node_translation = flat_translation_noise.view(N, num_nodes, 2)  # [N, 4, 2]
        per_node_rotation = flat_rotation_noise.view(N, num_nodes, 1)        # [N, 4, 1]  
        per_node_gripper = flat_gripper_noise.view(N, num_nodes, 1)          # [N, 4, 1]

        # get action noise
        predicted_noise = self.recover_se2_action_noise(per_node_translation, per_node_rotation, per_node_gripper)
        
        return predicted_noise, action_noise
    
    def recover_se2_action_noise(self, per_node_translation, per_node_rotation, per_node_gripper, agent_keypoints):
        """
        Recover SE(2) action noise from per-node adjustments using least squares alignment
        
        Args:
            per_node_translation: [N, 4, 2] - translation adjustments per keypoint
            per_node_rotation: [N, 4, 1] - rotation effect adjustments per keypoint  
            per_node_gripper: [N, 4, 1] - gripper state adjustments per keypoint
            agent_keypoints: [4, 2] - original keypoint positions relative to agent center
            
        Returns:
            action_noise: [N, 4] - SE(2) action noise [tx, ty, theta, gripper_state]
        """
        N, num_nodes, _ = per_node_translation.shape
        action_noise = torch.zeros(N, 4, device=self.device)
        
        for i in range(N):
            # Get the total displacement for each keypoint
            keypoint_displacements = per_node_translation[i] + per_node_rotation[i].expand(-1, 2)  # [4, 2]
            
            # Original keypoint positions (relative to agent center)
            original_keypoints = agent_keypoints  # [4, 2]
            
            # New keypoint positions after displacement
            new_keypoints = original_keypoints + keypoint_displacements  # [4, 2]
            
            # Solve for SE(2) transformation: new = R @ original + t
            # This is equivalent to the SVD step in the paper but for 2D
            se2_transform = self._solve_se2_alignment(original_keypoints, new_keypoints)
            
            # Extract translation and rotation from SE(2) matrix
            action_noise[i, 0] = se2_transform[0, 2]  # tx
            action_noise[i, 1] = se2_transform[1, 2]  # ty  
            action_noise[i, 2] = torch.atan2(se2_transform[1, 0], se2_transform[0, 0])  # theta
            
            # Gripper state: average across all keypoints
            action_noise[i, 3] = per_node_gripper[i].mean()
        
        return action_noise

    def _solve_se2_alignment(self, source_points, target_points):
        """
        Solve for SE(2) transformation that best aligns source to target points
        This is the 2D equivalent of SVD alignment in the paper
        
        Args:
            source_points: [4, 2] - original keypoint positions
            target_points: [4, 2] - displaced keypoint positions
            
        Returns:
            transform: [3, 3] - SE(2) transformation matrix
        """
        # Center the point sets
        source_centroid = source_points.mean(dim=0)  # [2]
        target_centroid = target_points.mean(dim=0)  # [2]
        
        source_centered = source_points - source_centroid  # [4, 2]
        target_centered = target_points - target_centroid  # [4, 2]
        
        # Compute cross-covariance matrix H = sum(source_i @ target_i^T)
        H = torch.mm(source_centered.T, target_centered)  # [2, 2]
        
        # SVD decomposition
        U, S, V = torch.svd(H)
        
        # Compute rotation matrix
        R = torch.mm(V, U.T)  # [2, 2]
        
        # Handle reflection case (ensure proper rotation)
        if torch.det(R) < 0:
            V[:, -1] *= -1
            R = torch.mm(V, U.T)
        
        # Compute translation
        t = target_centroid - torch.mv(R, source_centroid)  # [2]
        
        # Construct SE(2) matrix
        transform = torch.eye(3, device=self.device)
        transform[:2, :2] = R
        transform[:2, 2] = t
        
        return transform
    
    def _linear_beta_schedule(self, beta_start, beta_end, timesteps):
        """Create linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps, device = self.device)
    
    def _SE2_to_actions(self,se2):
        # actions from 3x3 matrix transformation => (x,y,theta)
        num_actions = se2.shape[0]
        actions = torch.zeros(num_actions, 3, device=self.device)
        
        # Extract translation components
        actions[:, 0] = se2[:, 0, 2]  # x_trans
        actions[:, 1] = se2[:, 1, 2]  # y_trans
        
        # Extract rotation angle and convert to degrees
        angles_rad = torch.atan2(se2[:, 1, 0], se2[:, 0, 0])
        actions[:, 2] = angles_rad
        
        return actions

    def _actions_to_SE2(self,actions):
        num_actions = actions.shape[0]
        se2_actions = torch.zeros(num_actions, 3,3, device=self.device) # 2x2 rot matr, with 2x1 translation mat
        angles = actions[:, 2]
        x_trans = actions[:, 0]
        y_trans = actions[:, 1]

        se2_actions[:, 0, 2] = x_trans
        se2_actions[:, 1, 2] = y_trans
        se2_actions[:, 2, 2] = 1

        _sin_theta = torch.sin(angles)
        _cos_theta = torch.cos(angles)
        se2_actions[:, 0, 0] = _cos_theta
        se2_actions[:, 0, 1] = - _sin_theta
        se2_actions[:, 1, 0] = _sin_theta
        se2_actions[:, 1, 1] = _cos_theta

        return se2_actions

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

    def _get_noisy_actions(self, clean_actions, timesteps, mode = 'large'):
        """
        Choose mode based on displacement magnitude
        """
        # # Compute displacement magnitude
        # translations = clean_actions[..., :2]
        # rotations = clean_actions[..., 2:3]
        
        # translation_magnitude = torch.norm(translations, dim=-1)
        # rotation_magnitude = torch.abs(rotations.squeeze(-1))
        
        # # Thresholds (tune these based on your domain)
        # translation_threshold = 0.05  # 5cm
        # rotation_threshold = 0.2      # ~11 degrees
        
        # large_displacement = (translation_magnitude > translation_threshold) | (rotation_magnitude > rotation_threshold)
        
        # if torch.any(large_displacement):
        if mode == 'large':
            return self._get_noisy_actions_large(clean_actions, timesteps)
        else:
            return self._get_noisy_actions_small(clean_actions, timesteps)
        
    def _get_noisy_actions_large(self, clean_actions, timesteps):
        """
        Large displacement: Add noise directly in SE(2) space with proper projection
        """
        binary_actions = clean_actions[..., 3:4]
        moving_actions = clean_actions[..., :-1]
        
        # Convert to SE(2) matrices
        SE2_clean = self._actions_to_SE2(moving_actions)  # [batch, 3, 3]
        
        # Sample noise in se(2) tangent space (this is key!)
        batch_size = SE2_clean.shape[0]
        se2_noise = torch.randn(batch_size, 3, device=self.device)  # [ρx, ρy, θ]
        
        # Scale noise for large displacements
        se2_noise[..., :2] *= 0.1  # Large translation noise (10cm)
        se2_noise[..., 2] *= 0.5   # Large rotation noise (≈30°)
        
        # Convert noise to SE(2) matrices using matrix exponential
        SE2_noise = self._se2_to_SE2(se2_noise)  # [batch, 3, 3]
        
        # Apply diffusion schedule
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1, 1)
        
        # For SE(2), we compose transformations: T_noisy = T_clean @ exp(√(1-α) * ξ)
        # Simplified as: T_noisy = √α * T_clean + √(1-α) * T_noise_scaled
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
        
        # Scale the noise transformation
        SE2_noise_scaled = torch.eye(3, device=self.device).unsqueeze(0) + sqrt_one_minus_alpha * (SE2_noise - torch.eye(3, device=self.device).unsqueeze(0))
        
        # Compose transformations (this is the proper SE(2) way)
        SE2_noisy = sqrt_alpha * SE2_clean + sqrt_one_minus_alpha * SE2_noise
        
        # Project back to SE(2) manifold (ensure rotation matrix constraints)
        SE2_noisy = self._project_to_SE2_manifold(SE2_noisy)
        
        # Convert back to actions
        noisy_moving_actions = self._SE2_to_actions(SE2_noisy)
        
        # Handle binary actions
        binary_noise = torch.randn_like(binary_actions)
        alpha_cumprod_binary = self.alpha_cumprod[timesteps].view(-1, 1)
        noisy_binary = torch.sqrt(alpha_cumprod_binary) * binary_actions + torch.sqrt(1 - alpha_cumprod_binary) * binary_noise
        
        # Combine results
        noisy_actions = torch.cat([noisy_moving_actions, noisy_binary], dim=-1)
        
        # The noise is the difference in action space (for training target)
        moving_noise = noisy_moving_actions - moving_actions
        full_noise = torch.cat([moving_noise, binary_noise], dim=-1)
        
        return noisy_actions, full_noise

    def _project_to_SE2_manifold(self, SE2_matrices):
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

    def _get_noisy_actions_small(self, clean_actions, timesteps):
        """
        Small displacement: se(2) tangent space approach (your current approach is mostly correct)
        """
        binary_actions = clean_actions[..., 3:4]
        moving_actions = clean_actions[..., :-1]
        
        # Convert to SE(2) then to se(2) tangent space
        SE2_actions = self._actions_to_SE2(moving_actions)
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
        noisy_binary = sqrt_alpha * binary_actions + sqrt_one_minus_alpha * binary_noise
        
        # Convert back: denormalize, se(2) → SE(2) → actions
        noisy_se2_unnorm = self._unnormalize_se2(noisy_se2_normalized)
        noisy_SE2 = self._se2_to_SE2(noisy_se2_unnorm)
        noisy_moving_actions = self._SE2_to_actions(noisy_SE2)
        
        # Combine results
        noisy_actions = torch.cat([noisy_moving_actions, noisy_binary], dim=-1)
        
        # Compute noise in action space (for training target)
        # This is important: noise should be in the same space as the network output
        clean_se2_unnorm = self._unnormalize_se2(se2_normalized)  # Without noise
        clean_SE2_reconstructed = self._se2_to_SE2(clean_se2_unnorm)
        clean_actions_reconstructed = self._SE2_to_actions(clean_SE2_reconstructed)
        
        moving_noise = noisy_moving_actions - clean_actions_reconstructed
        full_noise = torch.cat([moving_noise, binary_noise], dim=-1)
        
        return noisy_actions, full_noise
    
    def _normalise_se2(self,se2_actions):
        normalized = se2_actions.clone()
        normalized[...,:2] /= self.max_translation
        normalized[...,2:3] /= torch.pi 
        return normalized

    def _unnormalize_se2(self, normalized_se2):
        translation = normalized_se2[..., :2] * self.max_translation
        rotation = normalized_se2[..., 2:3] * torch.pi  
        return torch.cat([translation, rotation], dim=-1)
