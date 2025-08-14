import torch
import numpy as np
from tasks2d import LousyPacmanGameInterface as GameInterface
from tasks2d import LousyPacmanGameMode as GameMode
from tasks2d import LousyPacmanPseudoGameAction as Action
from tasks2d import LousyPacmanPlayerState as PlayerState
from configs import CONFIGS
from utils import _actions_to_SE2, _SE2_to_actions

def get_initial_keypoints(device):
    agent_kp = np.array([v for v in GameInterface.agent_keypoints.values()])
    return torch.tensor(agent_kp, device=device, dtype=torch.float)

def accumulate_actions(actions):
    """Accumulate SE(2) actions through matrix multiplication"""
    n = actions.shape[0]
    
    # Extract and reshape SE(2) matrices
    se2_matrices = actions[:, :9].view(n, 3, 3)
    state_actions = actions[:, 9:]
    
    # Compute cumulative matrix products
    cumulative_matrices = torch.zeros_like(se2_matrices)
    cumulative_matrices[0] = se2_matrices[0]
    
    for i in range(1, n):
        cumulative_matrices[i] = torch.matmul(cumulative_matrices[i-1], se2_matrices[i])
    
    # Flatten back and concatenate with state actions
    cumulative_se2_flat = cumulative_matrices.view(n, 9)
    cumulative_actions = torch.cat([cumulative_se2_flat, state_actions], dim=1)
    
    return cumulative_actions

def find_transformation_batch(source_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """
    Batch version: Find transformations for multiple point sets simultaneously.
    
    Args:
        source_points: Tensor of shape [B, N, 2] with B batches of N source points
        target_points: Tensor of shape [B, N, 2] with B batches of N target points
    
    Returns:
        Tensor of shape [B, 3] with columns [x_translation, y_translation, rotation_angle_radians]
    """
    batch_size = target_points.shape[0]
    
    # Precompute source centroid and centered points (constant across batches)
    source_centroid = torch.mean(source_points, dim=1)  # [B, 2] - Fixed: per batch centroid
    source_centered = source_points - source_centroid.unsqueeze(1)    # [B, N, 2]
    
    # Calculate target centroids for each batch
    target_centroids = torch.mean(target_points, dim=1)  # [B, 2]
    
    # Center target points for each batch
    target_centered = target_points - target_centroids.unsqueeze(1)  # [B, N, 2]
    
    # Calculate cross-covariance matrices for each batch
    H = torch.bmm(source_centered.transpose(1, 2), target_centered)  # [B, 2, 2]
    
    # Use SVD to find optimal rotations
    U, _, Vt = torch.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    
    # Handle reflections (ensure det(R) = 1)
    det_R = torch.det(R)
    reflection_mask = det_R < 0
    if reflection_mask.any():
        Vt_corrected = Vt.clone()
        Vt_corrected[reflection_mask, -1, :] *= -1
        R = torch.bmm(Vt_corrected.transpose(1, 2), U.transpose(1, 2))
    
    # Extract rotation angles
    rotation_angles = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    
    # Calculate translations
    translations = target_centroids - source_centroid  # [B, 2]
    x_translations = translations[:, 0]
    y_translations = translations[:, 1]
    
    # Stack transformations into [B, 3] tensor
    transformations = torch.stack([x_translations, y_translations, rotation_angles], dim=1)
    
    return transformations

def get_random_noisy_action(device, pred_horizon):
    """Generate initial noisy SE(2) actions for diffusion process"""
    max_rot = torch.tensor(CONFIGS['MAX_DEGREE_PER_TURN'], device=device)
    max_rot = torch.deg2rad(max_rot)
    
    # Generate pred_horizon random actions
    rot = (torch.rand(pred_horizon, device=device) - 0.5) * max_rot  # [-max_rot/2, max_rot/2]
    
    # Random translation for each action
    translation = torch.rand(pred_horizon, device=device) * CONFIGS['MAX_DISPLACEMENT_PER_STEP']
    x_translation = torch.cos(rot) * translation
    y_translation = torch.sin(rot) * translation
    
    # Random binary state for each action
    state_change = torch.rand(pred_horizon, device=device)
    
    # Create SE(2) transformation matrices for all actions
    cos_rot = torch.cos(rot)
    sin_rot = torch.sin(rot)
    
    se2_matrices = torch.zeros(pred_horizon, 3, 3, device=device)
    se2_matrices[:, 0, 0] = cos_rot
    se2_matrices[:, 0, 1] = -sin_rot
    se2_matrices[:, 0, 2] = x_translation
    se2_matrices[:, 1, 0] = sin_rot
    se2_matrices[:, 1, 1] = cos_rot
    se2_matrices[:, 1, 2] = y_translation
    se2_matrices[:, 2, 2] = 1.0
    
    # Flatten SE(2) matrices and append state changes
    flattened_se2 = se2_matrices.view(pred_horizon, -1)  # [pred_horizon, 9]
    se2_actions = torch.cat([flattened_se2, state_change.unsqueeze(-1)], dim=-1)  # [pred_horizon, 10]
    
    return se2_actions 

def se2_actions_to_gripper_positions(se2_actions, initial_gripper_positions):
    """
    Convert SE(2) actions to gripper node positions by applying transformations.
    
    Args:
        se2_actions: [pred_horizon, 10] - flattened SE(2) + state (ALREADY ACCUMULATED)
        initial_gripper_positions: [4, 2] - initial keypoint positions relative to gripper center
        
    Returns:
        gripper_positions: [pred_horizon, 4, 2] - transformed gripper positions
    """
    pred_horizon = se2_actions.shape[0]
    device = se2_actions.device
    
    # Extract SE(2) matrices (these are already accumulated cumulative transformations)
    se2_matrices = se2_actions[:, :-1].view(pred_horizon, 3, 3)  # [T, 3, 3]
    
    # Apply transformations to initial gripper positions
    # Add homogeneous coordinate
    initial_homo = torch.cat([
        initial_gripper_positions, 
        torch.ones(4, 1, device=device)
    ], dim=1)  # [4, 3]
    
    # Transform positions for each timestep
    gripper_positions = torch.zeros(pred_horizon, 4, 2, device=device)
    
    for t in range(pred_horizon):
        # Apply cumulative transformation: T_cumulative @ points^T
        transformed = se2_matrices[t] @ initial_homo.T  # [3, 4]
        gripper_positions[t] = transformed[:2].T  # [4, 2]
    
    return gripper_positions

def gripper_positions_to_se2_actions(old_positions, new_positions, device):
    """
    Extract SE(2) transformations from gripper position changes.
    
    Args:
        old_positions: [pred_horizon, 4, 2]
        new_positions: [pred_horizon, 4, 2]
        
    Returns:
        se2_actions: [pred_horizon, 9] - SE(2) transformations
    """
    pred_horizon = old_positions.shape[0]
    se2_actions = torch.zeros(pred_horizon, 9, device=device)
    
    for t in range(pred_horizon):
        # Find transformation between old and new positions
        transformation = find_transformation_batch(
            old_positions[t:t+1], 
            new_positions[t:t+1]
        )  # [1, 3] -> [x_trans, y_trans, rotation]
        
        # Convert to SE(2) matrix
        x_trans, y_trans, rotation = transformation[0]
        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)
        
        se2_matrix = torch.zeros(3, 3, device=device)
        se2_matrix[0, 0] = cos_rot
        se2_matrix[0, 1] = -sin_rot
        se2_matrix[0, 2] = x_trans
        se2_matrix[1, 0] = sin_rot
        se2_matrix[1, 1] = cos_rot
        se2_matrix[1, 2] = y_trans
        se2_matrix[2, 2] = 1.0
        
        se2_actions[t] = se2_matrix.flatten()
    
    return se2_actions

def ddim_step_corrected(alpha_cumprod, noisy_gripper_positions, predicted_denoising_directions, 
                       timestep, timestep_prev):
    """
    Corrected DDIM step that operates on gripper positions.
    
    Args:
        alpha_cumprod: Alpha cumulative product schedule
        noisy_gripper_positions: [T, 4, 2] - Current noisy gripper positions
        predicted_denoising_directions: [T, 4, 5] - Per-node denoising directions
        timestep: Current timestep k
        timestep_prev: Previous timestep k-1
        
    Returns:
        denoised_positions: [T, 4, 2] - Denoised gripper positions
    """
    device = alpha_cumprod.device
    
    # Get alpha values
    alpha_cumprod_t = alpha_cumprod[timestep]
    if timestep_prev >= 0:
        alpha_cumprod_t_prev = alpha_cumprod[timestep_prev]
    else:
        alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
    
    # Step 1: Convert predicted denoising directions to position updates
    # Combine translation and rotation components: [x_t + rx_t, y_t + ry_t]
    x_combined = predicted_denoising_directions[..., 0] + predicted_denoising_directions[..., 2]
    y_combined = predicted_denoising_directions[..., 1] + predicted_denoising_directions[..., 3]
    position_denoising = torch.stack([x_combined, y_combined], dim=-1)  # [T, 4, 2]
    
    # Step 2: Predict clean positions (x_0 prediction)
    predicted_clean_positions = (noisy_gripper_positions - torch.sqrt(1 - alpha_cumprod_t) * position_denoising) / torch.sqrt(alpha_cumprod_t)
    
    # Step 3: DDIM step on positions
    denoised_positions = (
        torch.sqrt(alpha_cumprod_t_prev) * predicted_clean_positions +
        torch.sqrt(1 - alpha_cumprod_t_prev) * position_denoising
    )
    
    return denoised_positions

def corrected_evaluation_loop(curr_obs, provided_demos, device, agent, num_diffusion_steps, 
                             num_inference_steps, agent_keypoints):
    """Corrected evaluation loop with proper DDIM on gripper positions."""
    
    # Initialize with random noise actions and accumulate them
    noisy_actions = get_random_noisy_action(device, agent.policy.pred_horizon)
    noisy_actions = accumulate_actions(noisy_actions)  # Accumulate for cumulative positions
    
    # Convert initial actions to gripper positions
    current_gripper_positions = se2_actions_to_gripper_positions(
        noisy_actions, agent_keypoints
    )  # [T, 4, 2]
    
    # Store initial noise for debugging
    initial_noisy_actions = noisy_actions.clone()
    
    # DDIM denoising loop
    timesteps = torch.linspace(num_diffusion_steps-1, 0, num_inference_steps, dtype=torch.long, device=device)
    
    for i, timestep in enumerate(timesteps):
        current_timestep = timesteps[i]
        next_timestep = timesteps[i+1] if i < len(timesteps)-1 else -1
        
        # Get noise prediction from agent (per-node denoising directions)
        predicted_node_noise = agent.predict(
            curr_obs,
            provided_demos,
            noisy_actions,  # Still pass actions for context
        )  # [T, 4, 5]
        
        # Apply DDIM step on gripper positions
        current_gripper_positions = ddim_step_corrected(
            agent.alpha_cumprod,
            current_gripper_positions,
            predicted_node_noise,
            current_timestep,
            next_timestep
        )
        
        # Convert back to SE(2) actions for next iteration
        initial_positions = agent_keypoints.unsqueeze(0).expand(
            current_gripper_positions.shape[0], -1, -1
        )  # [T, 4, 2]
        
        denoised_se2 = gripper_positions_to_se2_actions(
            initial_positions,
            current_gripper_positions,
            device
        )
        
        # Handle state changes
        state_changes = predicted_node_noise[..., 4].mean(dim=-1, keepdim=True)  # [T, 1]
        
        # Combine SE(2) and state
        noisy_actions = torch.cat([denoised_se2, state_changes], dim=-1)  # [T, 10]
    
    # Extract final clean actions
    final_actions = se2_action_sequence_to_obj_list(noisy_actions)
    return final_actions, initial_noisy_actions

def se2_action_sequence_to_obj_list(se2_action_sequence):
    """Convert flattened SE(2) action sequence to list of game action objects"""
    # Input shape: [pred_horizon, 10]
    action_objects = []

    for i in range(se2_action_sequence.shape[0]):  # Iterate over pred_horizon
        action = se2_action_sequence[i]  # [10]
        
        # Reshape flattened SE(2) back to 3x3 matrix
        se2_matrix = action[:-1].view(3, 3)  # First 9 elements -> [3, 3]
        state_change = action[-1]  # Last element
        
        # Extract translation components from SE(2) matrix
        x_trans = se2_matrix[0, 2].item()
        y_trans = se2_matrix[1, 2].item()
        
        # Extract rotation from SE(2) matrix
        cos_rot = se2_matrix[0, 0].item()
        sin_rot = se2_matrix[1, 0].item()
        rotation = torch.atan2(torch.tensor(sin_rot), torch.tensor(cos_rot)).item()
        
        # Convert to movement magnitude and rotation in degrees
        forward_movement = (x_trans ** 2 + y_trans ** 2) ** 0.5
        rotation_deg = np.rad2deg(rotation)
        
        # Convert binary state to PlayerState
        player_state = PlayerState.EATING if state_change.item() > 0.5 else PlayerState.NOT_EATING
        
        action_obj = Action(
            forward_movement=forward_movement,
            rotation_deg=rotation_deg,
            state_change=player_state
        )
        
        action_objects.append(action_obj)
    
    return action_objects

def collect_demos(game_interface, num_demos, manual=CONFIGS['MANUAL_DEMO_COLLECT']):
    provided_demos = []
    sampling_rate = CONFIGS['TEST_SAMPLING_RATE']

    print(f"Collecting {num_demos} demonstrations...")
    if manual:
        for n in range(num_demos):
            game_interface.start_game()
            while game_interface.running:
                game_interface.step()
            
            # Process demo to extract relevant observations
            demo = game_interface.observations
            if len(demo) > 1:
                indices = np.linspace(0, len(demo)-1, min(len(demo), sampling_rate), dtype=int)
                processed_demo = [demo[i] for i in indices]
                provided_demos.append(processed_demo)
            
            game_interface.reset()
        return provided_demos
    
    print(f"Collected {len(provided_demos)} demonstrations")
    pass

if __name__ == "__main__":
    from configs import CONFIGS, version, _type, geo_version
    from agent_files import GeometryEncoder2D, full_train, initialise_geometry_encoder, InstantPolicyAgent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train geometry encoder    
    geometry_encoder_filename = f'geometry_encoder_2d_v{geo_version}.pth'
    node_embd_dim = CONFIGS['NUM_ATT_HEADS'] * CONFIGS['HEAD_DIM']
    grouping_radius = CONFIGS['GROUPING_RADIUS']
    num_sampled_pc = CONFIGS['NUM_SAMPLED_POINTCLOUDS']

    model = GeometryEncoder2D(num_centers=num_sampled_pc, radius=grouping_radius, 
                             node_embd_dim=node_embd_dim, device=device).to(device)
    geometry_encoder = initialise_geometry_encoder(model, geometry_encoder_filename, device=device)

    agent = InstantPolicyAgent(
        device=device, 
        geometry_encoder=geometry_encoder,
        max_translation=CONFIGS['TESTING_MAX_UNIT_TRANSLATION'],
        max_rotation=CONFIGS['MAX_ROTATION_DEG'],
        num_diffusion_steps=CONFIGS['NUM_DIFFUSION_STEPS'],
        num_agent_nodes=CONFIGS['NUM_AGENT_NODES'],
        pred_horizon=CONFIGS['PRED_HORIZON'],
        num_att_heads=CONFIGS['NUM_ATT_HEADS'],
        head_dim=CONFIGS['HEAD_DIM'],
        agent_state_embd_dim=CONFIGS['AGENT_STATE_EMB_DIM'],
        edge_pos_dim=CONFIGS['EDGE_POS_DIM'],
        max_flow_translation=CONFIGS['MAX_DISPLACEMENT_PER_STEP'] * 2,
        max_flow_rotation=CONFIGS['MAX_DEGREE_PER_TURN'] * 2
    )
    
    # Load the trained agent
    if torch.cuda.is_available() and device == 'cuda':
        state_dict = torch.load(CONFIGS['MODEL_FILE_PATH'], map_location='cuda')
    else:
        state_dict = torch.load(CONFIGS['MODEL_FILE_PATH'], map_location='cpu')
    
    agent.load_state_dict(state_dict)
    agent.eval()
    agent.to(device)
    
    # Collect demonstration data
    game_interface = GameInterface(
        mode=GameMode.DEMO_MODE,
        num_sampled_points=CONFIGS['NUM_SAMPLED_POINTCLOUDS']
    )
    num_demos_given = CONFIGS['TEST_NUM_DEMO_GIVEN']
    provided_demos = collect_demos(game_interface=game_interface, num_demos=num_demos_given, manual=True)

    # Start evaluation
    num_diffusion_steps = agent.num_diffusion_steps
    game_interface.change_mode(GameMode.AGENT_MODE)
    
    t = 0
    curr_obs = game_interface.start_game()
    done = False
    num_inference_steps = 50
    
    print("Starting evaluation...")
    AGENT_KEYPOINTS = get_initial_keypoints(device=device)
    
    with torch.no_grad(): 
        while t < CONFIGS['MAX_INFERENCE_ITER'] and not done:
            final_actions, initial_noisy = corrected_evaluation_loop(
                curr_obs=curr_obs, 
                provided_demos=provided_demos,
                device=device, 
                agent=agent, 
                num_diffusion_steps=num_diffusion_steps,
                num_inference_steps=num_inference_steps,
                agent_keypoints=AGENT_KEYPOINTS
            )
            
            # Execute each action in the sequence
            for i, action_obj in enumerate(final_actions):
                if done:
                    break

                # Debugging output
                print(f'Step {t}: Action - Movement: {action_obj.forward_movement:.3f}, '
                      f'Rotation: {action_obj.rotation:.1f}°, State: {action_obj.state_change}')
                
                curr_obs = game_interface.step(action_obj)
                done = curr_obs.get('done', False)
                t += 1
                
                if done:
                    break
    
    if game_interface.game.game_won:
        print('Won Game!')
    else:
        print('Nice Try!')
    
    print(f"Evaluation completed after {t} steps")