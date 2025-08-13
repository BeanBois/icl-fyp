import torch
import numpy as np
from tasks2d import LousyPacmanGameInterface as GameInterface
from tasks2d import LousyPacmanGameMode as GameMode
from tasks2d import LousyPacmanPseudoGameAction as Action
from tasks2d import LousyPacmanPlayerState as PlayerState
from configs import CONFIGS, action_mode
from utils import _actions_to_SE2

# NOT FIXED

# Need to change actions to SE2 

def get_random_noisy_action(device, pred_horizon):
    """Generate initial noisy SE(2) actions for diffusion process"""
    max_rot = torch.tensor(CONFIGS['TESTING_MAX_UNIT_ROTATION'], device=device)
    max_rot = torch.deg2rad(max_rot)
    
    # Generate pred_horizon random actions
    rot = (torch.rand(pred_horizon, device=device) - 0.5) * 2 * max_rot  # [-max_rot, max_rot]
    
    # Random translation for each action
    translation = torch.rand(pred_horizon, device=device) * CONFIGS['TESTING_MAX_UNIT_TRANSLATION']
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

def accumulate_actions(actions):
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


def ddim_step(alpha_cumprod, noisy_se2_actions, predicted_per_node_denoising_direction, timestep, timestep_prev):
    """Corrected DDIM step with proper indexing for flattened SE(2) actions"""
    # Get alpha values with proper bounds checking
    alpha_cumprod_t = alpha_cumprod[timestep]
    if timestep_prev >= 0:
        alpha_cumprod_t_prev = alpha_cumprod[timestep_prev]
    else:
        alpha_cumprod_t_prev = torch.tensor(1.0, device=alpha_cumprod.device)
    
    # Convert predicted_per_node_denoising_direction [N, 4, 5] to SE(2) format
    # Input format: {x_t, y_t, r_tx, r_ty, state_change}
    batch_size = predicted_per_node_denoising_direction.shape[0]
    pred_horizon = predicted_per_node_denoising_direction.shape[1]  # Should be 4
    device = noisy_se2_actions.device
    
    # Extract components from predicted noise [N, 4, 5]
    x_trans = predicted_per_node_denoising_direction[..., 0]      # [N, 4]
    y_trans = predicted_per_node_denoising_direction[..., 1]      # [N, 4] 
    r_tx = predicted_per_node_denoising_direction[..., 2]         # [N, 4]
    r_ty = predicted_per_node_denoising_direction[..., 3]         # [N, 4]
    state_change = predicted_per_node_denoising_direction[..., 4] # [N, 4]
    
    # Convert r_tx, r_ty to rotation angle
    rotation = torch.atan2(r_ty, r_tx)  # [N, 4]
    
    # Create SE(2) matrices from the denoising direction
    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)
    
    # Create full SE(2) noise matrices [N, 4, 9]
    se2_noise = torch.zeros(batch_size, pred_horizon, 9, device=device)
    se2_noise[..., 0] = cos_rot    # [0,0] - cos(θ)
    se2_noise[..., 1] = -sin_rot   # [0,1] - -sin(θ)
    se2_noise[..., 2] = x_trans    # [0,2] - x translation
    se2_noise[..., 3] = sin_rot    # [1,0] - sin(θ)
    se2_noise[..., 4] = cos_rot    # [1,1] - cos(θ)
    se2_noise[..., 5] = y_trans    # [1,2] - y translation
    se2_noise[..., 6] = 0.0        # [2,0] - always 0
    se2_noise[..., 7] = 0.0        # [2,1] - always 0
    se2_noise[..., 8] = 0.0        # [2,2] - always 1, but no noise
    
    # Concatenate SE(2) noise with state noise [N, 4, 10]
    predicted_noise = torch.cat([se2_noise, state_change.unsqueeze(-1)], dim=-1)
    
    # Predict clean actions p̂₀ (estimated x₀)
    # p̂₀ = (p_k - √(1-α_k) * ε) / √α_k
    predicted_clean = (
        noisy_se2_actions - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
    ) / torch.sqrt(alpha_cumprod_t)
    
    # DDIM update following the formula:
    # p_{k-1} = √α_{k-1} * p̂₀ + √(1-α_{k-1}) * (p_k - √α_k * p̂₀) / √(1-α_k)
    if timestep_prev < 0:  # Final step
        return predicted_clean
    else:
        # Direction pointing from x_t to x_0
        direction = (noisy_se2_actions - torch.sqrt(alpha_cumprod_t) * predicted_clean) / torch.sqrt(1 - alpha_cumprod_t)
        
        # DDIM step: p_{k-1} = √α_{k-1} * p̂₀ + √(1-α_{k-1}) * direction
        denoised_actions = (
            torch.sqrt(alpha_cumprod_t_prev) * predicted_clean +
            torch.sqrt(1 - alpha_cumprod_t_prev) * direction
        )
        return denoised_actions

def retreive_actions(predicted_per_node_noises):
    pass 

def se2_action_to_obj(se2_action):
    """Convert flattened SE(2) action to game action object"""
    action = se2_action.squeeze(0)  # Remove batch dimension
    
    # Reshape flattened SE(2) back to 3x3 matrix
    se2_matrix = action[:-1].view(3, 3)  # First 9 elements
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
    
    return action_obj


def set_device_recursive(obj, device):
    """Recursively set device for all NN components"""
    if hasattr(obj, 'device'):
        obj.device = device
    
    # Handle nn.Module components
    if hasattr(obj, 'to'):
        obj.to(device)
    
    # Recursively check all attributes
    for attr_name in dir(obj):
        if not attr_name.startswith('_'):  # Skip private attributes
            try:
                attr = getattr(obj, attr_name)
                if hasattr(attr, '__dict__'):  # Has attributes to explore
                    set_device_recursive(attr, device)
            except:
                continue

if __name__ == "__main__":
    from configs import CONFIGS
    import agent_files
    # train geometry encoder first

    manual = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the trained agent
    if str(device) == 'cpu':
        agent = torch.load(CONFIGS['MODEL_FILE_PATH'], weights_only=False, map_location=torch.device('cpu'))
        set_device_recursive(agent, 'cpu')
    
    else:
        agent = torch.load(CONFIGS['MODEL_FILE_PATH'], weights_only=False)
    
    agent.eval()
    agent.to(device)
    
    # Collect demonstration data
    game_interface = GameInterface(
        mode=GameMode.DEMO_MODE,
        num_sampled_points=CONFIGS['NUM_SAMPLED_POINTCLOUDS']
    )
    num_demos_given = CONFIGS['TEST_NUM_DEMO_GIVEN']
    sampling_rate = CONFIGS['SAMPLING_RATE']
    provided_demos = []
    
    # i want to automate these such that its like yk automated
    print(f"Collecting {num_demos_given} demonstrations...")
    if manual:
        for n in range(num_demos_given):
            game_interface.start_game()
            step_count = 0
            while game_interface.running:
                game_interface.step()
            
            # Process demo to extract relevant observations
            # Take evenly spaced samples from the demo
            demo = game_interface.observations
            if len(demo) > 1:
                indices = np.linspace(0, len(demo)-1, min(len(demo), sampling_rate), dtype=int)
                processed_demo = [demo[i] for i in indices]
                provided_demos.append(processed_demo)
            
            game_interface.reset()
    else: # ignore this for now
        # choose a config
        # load config with game_interface.load_config()
        # sample from demoset num_demo_given
        # downsample with sampling_rate
        pass 
        
    
    print(f"Collected {len(provided_demos)} demonstrations")
    
    # Start evaluation
    num_diffusion_steps = agent.num_diffusion_steps  # Use agent's diffusion steps
    game_interface.change_mode(GameMode.AGENT_MODE)
    
    # Prepare alpha schedules for DDIM
    alpha_cumprod_prev = torch.cat([
        torch.tensor([1.0], device=device),
        agent.alpha_cumprod[:-1]
    ])
    
    t = 0
    curr_obs = game_interface.start_game()
    done = False
    
    print("Starting evaluation...")
    
    with torch.no_grad(): 
        prev_actions = torch.zeros(4, device = device)
        while t < CONFIGS['MAX_INFERENCE_ITER'] and not done:
            # Initialize with random noise
            noisy_actions = get_random_noisy_action(device, agent.policy.pred_horizon)
            acc_noisy_actions = accumulate_actions(noisy_actions)
            print(f'initialy noisy action: {noisy_actions}')
            
            # DDIM denoising loop (reverse diffusion)
            num_inference_steps = 50
            timesteps = torch.linspace(num_diffusion_steps-1, 0, num_inference_steps, dtype=torch.long, device=device)
            for i, timestep in enumerate(timesteps):
                current_timestep = timesteps[i]
                next_timestep = timesteps[i+1] if i < len(timesteps)-1 else -1
                
                # Get noise prediction from agent
                predicted_noise, _ = agent(
                    curr_obs,
                    provided_demos,
                    acc_noisy_actions,
                    action_mode
                )
                
                # DDIM denoising step
                acc_noisy_actions = ddim_step(
                    agent.alpha_cumprod,
                    acc_noisy_actions,
                    predicted_noise,
                    current_timestep,
                    next_timestep
                )


                
            # Convert final denoised action to game action
            action_objs = se2_action_to_obj(acc_noisy_actions)
            for action_obj in action_objs:
                print(f'Action: {action_obj.forward_movement, action_obj.rotation, action_obj.state_change}')
                curr_obs = game_interface.step(action_obj)
                done = curr_obs.get('done', False)
                t += 1
                if done:
                    break 
            if done:
                break 
    
    
    if game_interface.game.game_won:
        print('Won Game')
    else:
        print('Nice Try')
    
    print(f"Evaluation completed after {t} steps")