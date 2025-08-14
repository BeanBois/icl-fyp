import torch
import numpy as np
from tasks2d import LousyPacmanGameInterface as GameInterface
from tasks2d import LousyPacmanGameMode as GameMode
from tasks2d import LousyPacmanPseudoGameAction as Action
from tasks2d import LousyPacmanPlayerState as PlayerState
from configs import CONFIGS
from utils import _actions_to_SE2, _SE2_to_actions


# TODO: we cant just update 2 differnet units (pixel and rad) during denoising ? 
# During training, we learn to predict per-node denoising dir given noisy actions 
# 

# might have to resort to paper implementation and get predicted node positions 
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
    source_centroid = torch.mean(source_points, dim=0)  # [2]
    source_centered = source_points - source_centroid    # [N, 2]
    
    # Calculate target centroids for each batch
    target_centroids = torch.mean(target_points, dim=1)  # [B, 2]
    
    # Center target points for each batch
    target_centered = target_points - target_centroids.unsqueeze(1)  # [B, N, 2]
    
    # Calculate cross-covariance matrices for each batch
    # source_centered: [N, 2] -> [2, N], target_centered: [B, N, 2]
    H = torch.bmm(source_centered.T.unsqueeze(0).expand(batch_size, -1, -1), target_centered)  # [B, 2, 2]
    
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
    
    # Calculate translations (target_centroid - source_centroid for each batch)
    translations = target_centroids - source_centroid.unsqueeze(0)  # [B, 2]
    x_translations = translations[:, 0]
    y_translations = translations[:, 1]
    
    # Stack transformations into [B, 3] tensor
    transformations = torch.stack([x_translations, y_translations, rotation_angles], dim=1)
    
    return transformations

def deduct_se2(main, subtractor, device):
    num_elements = main.shape[0]
    main = main.view(-1,3,3)
    subtractor = subtractor.view(-1,3,3)

    results = torch.zeros((num_elements, 3,3),device=device)
    results[:,0,2] = main[:,0,2] - subtractor[:,0,2]  # clamp here if wanna 
    results[:,1,2] = main[:,1,2] - subtractor[:,1,2] 
    #now calculate angle
    main_theta_rads = torch.atan2(main[:, 1, 0], main[:, 0, 0])
    sub_theta_rads = torch.atan2(subtractor[:, 1, 0], subtractor[:, 0, 0])
    diff = main_theta_rads - sub_theta_rads # clamp here if u wanna
    final_theta = torch.atan2(torch.sin(diff), torch.cos(diff))

    _sin_theta = torch.sin(final_theta)
    _cos_theta = torch.cos(final_theta)
    results[:, 0, 0] = _cos_theta
    results[:, 0, 1] = - _sin_theta
    results[:, 1, 0] = _sin_theta
    results[:, 1, 1] = _cos_theta


    return results.view(num_elements,-1)

def deduct_actions(main, subtractor,device):
    se_main = main[:, :-1]
    se_sub = subtractor[:, :-1]
    se_results = deduct_se2(se_main, se_sub, device)

    state_main = main[:,-1]
    state_sub = subtractor[:,-1]
    state_results = torch.clamp(state_main-state_sub, -1, 1)
    return torch.cat([se_results, state_results.view(-1,1)], dim=-1)

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
    # SE(2) matrix is 3x3: [[cos(θ) -sin(θ) x]
    #                       [sin(θ)  cos(θ) y]
    #                       [0       0      1]]
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
    # Shape: [pred_horizon, 9 + 1] = [pred_horizon, 10]
    flattened_se2 = se2_matrices.view(pred_horizon, -1)  # [pred_horizon, 9]
    se2_actions = torch.cat([flattened_se2, state_change.unsqueeze(-1)], dim=-1)  # [pred_horizon, 10]
    
    return se2_actions 

def predicted_per_node_denoising_direction_to_action_denoise_dir(device, predicted_per_node_denoising_direction, initial_node_positions):
    """
    Convert predicted per-node denoising directions to action denoising directions.
    
    Steps:
    1. Reduce [pred_horizon, 4, 5] to [pred_horizon, 4, 2] by: [x_t + rx_t, y_t + ry_t]
    2. For each timestep, find 1 singular SE(2) action that produces those 4 node movements
    3. Average state changes over 4 nodes for each timestep
    
    Args:
        predicted_per_node_denoising_direction: [N, pred_horizon, 4, 5] tensor {x_t, y_t, r_tx, r_ty, state_change}
        initial_node_positions: [4, 2] tensor - node positions relative to center
            
    Returns:
        action_denoise_dir: [N, 10] tensor (flattened SE(2) + state format)
    """
    batch_size = predicted_per_node_denoising_direction.shape[0]
    pred_horizon = predicted_per_node_denoising_direction.shape[1]  # Should be 4
    device = predicted_per_node_denoising_direction.device
    
    # Step 1: Reduce [4, 5] to [4, 2] for each timestep
    x_combined = predicted_per_node_denoising_direction[..., 0] + predicted_per_node_denoising_direction[..., 2]  # [N, 4, 4]
    y_combined = predicted_per_node_denoising_direction[..., 1] + predicted_per_node_denoising_direction[..., 3]  # [N, 4, 4]
    combined_movements = torch.stack([x_combined, y_combined], dim=-1)  # [N, 4, 2]
    
    # Step 2: Find denoising moving actions
    denoising_moving_action = find_transformation_batch(initial_node_positions, combined_movements) # [B,3]
    se2_denoising_moving_action = _actions_to_SE2(denoising_moving_action, device).view(batch_size,-1) # [N, 9]
    
    # Step 3: Average state changes over 4 nodes for each timestep
    state_changes_noise = predicted_per_node_denoising_direction[:, :, 4] # [N,4,1]
    state_change_agg = torch.mean(state_changes_noise, dim = 1) # [N,1]


    return torch.cat([se2_denoising_moving_action,state_change_agg.view(-1,1)], dim=-1)  # [N, 10]



def ddim_step(alpha_cumprod, noisy_se2_actions, denoising_action, timestep, timestep_prev):
    """
    DDIM step for denoising SE(2) actions using pre-computed action denoising directions.
    
    Args:
        alpha_cumprod: Alpha cumulative product schedule
        noisy_se2_actions: [N, 10] - Current noisy actions (flattened SE(2) + state)
        denoising_action: [N, 10] - Action denoising directions
        timestep: Current timestep k
        timestep_prev: Previous timestep k-1
        
    Returns:
        denoised_actions: [N, 10] - Denoised actions at timestep k-1
    """
    # Get alpha values with proper bounds checking
    alpha_cumprod_t = alpha_cumprod[timestep]
    if timestep_prev >= 0:
        alpha_cumprod_t_prev = alpha_cumprod[timestep_prev]
    else:
        alpha_cumprod_t_prev = torch.tensor(1.0, device=alpha_cumprod.device)
    
    # needs to be handleed better 
    # both are in se2 space, bit flattened


    predicted_denoised_actions = deduct_actions( main=noisy_se2_actions, subtractor=denoising_action, device=alpha_cumprod.device)

    # maybe do some clamping

    # our ddim step is :
        # 
        #denoised_actions = sqrt(alpha_cumprod_t_prev) * (noisy_se2_actions - denoisning_action)
    denoised_actions = torch.sqrt(alpha_cumprod_t_prev) * predicted_denoised_actions + \
                        torch.sqrt((1-alpha_cumprod_t_prev)/(1-alpha_cumprod_t)) * (noisy_se2_actions - torch.sqrt(alpha_cumprod_t) * predicted_denoised_actions)
    return denoised_actions


def se2_action_sequence_to_obj_list(se2_action_sequence):
    """Convert flattened SE(2) action sequence to list of game action objects"""
    # Input shape: [batch_size, pred_horizon, 10]
    action_sequence = se2_action_sequence.squeeze(0)  # Remove batch dimension -> [pred_horizon, 10]
    action_objects = []

    for i in range(action_sequence.shape[0]):  # Iterate over pred_horizon
        action = action_sequence[i]  # [10]
        
        # Reshape flattened SE(2) back to 3x3 matrix
        se2_matrix = action[:-1].view(3, 3)  # First 9 elements -> [3, 3]
        state_change = action[-1]  # Last element
        
        # Extract translation components from SE(2) matrix
        x_trans = se2_matrix[0, 2].item()
        y_trans = se2_matrix[1, 2].item()
        
        # Extract rotation from SE(2) matrix
        # rotation = atan2(sin(θ), cos(θ))
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

def collect_demos(game_interface, num_demos, manual = CONFIGS['MANUAL_DEMO_COLLECT']):
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

        # Add normalization parameters
        max_flow_translation=CONFIGS['MAX_DISPLACEMENT_PER_STEP'] * 2,  # 2cm = twice the 1cm max displacement
        max_flow_rotation=CONFIGS['MAX_DEGREE_PER_TURN'] * 2      # 6 degrees = twice the 3 degree max displacement
    )
    # Load the trained agent
    if torch.cuda.is_available() and device == 'cuda':
        state_dict = torch.load(CONFIGS['MODEL_FILE_PATH'], map_location='cuda')
    else:
        state_dict = torch.load(CONFIGS['MODEL_FILE_PATH'], map_location='cpu')
    
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
    
    print("Starting evaluation...")
    AGENT_KEYPOINTS = get_initial_keypoints(device=device)
    with torch.no_grad(): 
        while t < CONFIGS['MAX_INFERENCE_ITER'] and not done:
            # Initialize with random noise - add batch dimension
            noisy_actions = get_random_noisy_action(device, agent.policy.pred_horizon)
            temp = noisy_actions # debugging 
            # noisy_actions = noisy_actions.unsqueeze(0)  # Add batch dimension [1, pred_horizon, 10]
            
            # DDIM denoising loop (reverse diffusion)
            num_inference_steps = 50
            timesteps = torch.linspace(num_diffusion_steps-1, 0, num_inference_steps, dtype=torch.long, device=device)
            
            for i, timestep in enumerate(timesteps):
                current_timestep = timesteps[i]
                next_timestep = timesteps[i+1] if i < len(timesteps)-1 else -1
                
                # Get noise prediction from agent
                # The agent should return per-node denoising directions [batch_size, pred_horizon, 5]
                predicted_node_noise = agent.predict(
                    curr_obs,
                    provided_demos,
                    noisy_actions,  # Pass current noisy actions
                )

                denoising_action = predicted_per_node_denoising_direction_to_action_denoise_dir(device,predicted_node_noise, AGENT_KEYPOINTS)

                # breakpoint()
                # update the positions of agent nodes by taking a denoising step according to the DDIM (Song et al., 2020): 
                # since noisy actions are used to transform agent node positions, denoising the noisy actions will be the equivalent of
                # doing this. So denoise actions using the predicted-per-node-dir with DDIM denoising step. 
                # First, 
                noisy_actions = ddim_step(
                    agent.alpha_cumprod,
                    noisy_actions,
                    denoising_action,
                    current_timestep,
                    next_timestep
                )
            
            # Extract final clean actions and execute them
            final_actions = se2_action_sequence_to_obj_list(noisy_actions)
            # Execute each action in the sequence
            for i,action_obj in enumerate(final_actions):
                if done:
                    break

                # for debuggin
                noisy_action = temp[i]
                se2_noised = noisy_action[:-1]
                action = _SE2_to_actions(se2_noised.view(1,3,3),device)[0]
                noised_forward_movement = (action[0]**2 + action[1]**2) **0.5
                noised_rot = torch.rad2deg(action[2])
                state_noised = noisy_action[-1]

                print(f'Noisy action - Movement: {noised_forward_movement:.3f}, '
                      f'Rotation: {noised_rot:.1f}°, State: {state_noised}')
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