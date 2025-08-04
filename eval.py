import torch
import numpy as np
from tasks2d import LousyPacmanGameInterface as GameInterface
from tasks2d import LousyPacmanGameMode as GameMode
from tasks2d import LousyPacmanPseudoGameAction as Action
from tasks2d import LousyPacmanPlayerState as PlayerState
from configs import CONFIGS

def get_random_noisy_action(device):
    """Generate initial noisy action for diffusion process"""
    max_rot = torch.tensor(CONFIGS['MAX_ROTATION_DEG'], device=device)
    max_rot = torch.deg2rad(max_rot)
    rot = (torch.rand(1, device=device) - 0.5) * 2 * max_rot  # [-max_rot, max_rot]
    
    # Random translation
    translations = torch.randn(2, device=device) * CONFIGS['TESTING_MAX_UNIT_TRANSLATION']
    
    # Random binary state
    state_change = torch.randn(1, device=device)
    
    # Combine into 4D action: [x_trans, y_trans, rotation, state_change]
    actions = torch.cat([translations, rot, state_change], dim=-1)
    return actions.unsqueeze(0)  # Add batch dimension

def ddim_step(alpha_cumprod, alpha_cumprod_prev, noisy_actions, predicted_noise, timestep):
    """DDIM denoising step (used in inference)"""
    alpha_cumprod_t = alpha_cumprod[timestep]
    alpha_cumprod_t_prev = alpha_cumprod_prev[timestep]
    
    # Predict clean actions
    predicted_clean = (
        noisy_actions - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
    ) / torch.sqrt(alpha_cumprod_t)
    
    # DDIM update
    denoised_actions = (
        torch.sqrt(alpha_cumprod_t_prev) * predicted_clean +
        torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
    )
    
    return denoised_actions

def action_tensor_to_obj(action_tensor):
    """Convert action tensor to game action object"""
    action = action_tensor.squeeze(0)  # Remove batch dimension
    
    # Extract components
    x_trans = action[0].item()
    y_trans = action[1].item()
    rotation = action[2].item()
    state_change = action[3].item()
    
    # Convert to movement magnitude and rotation in degrees
    forward_movement = (x_trans ** 2 + y_trans ** 2) ** 0.5
    rotation_deg = np.rad2deg(rotation)
    
    # Convert binary state to PlayerState
    player_state = PlayerState.EATING if state_change > 0 else PlayerState.NOT_EATING
    
    action_obj = Action(
        forward_movement=forward_movement,
        rotation=rotation_deg,
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
    else:
        # choose a config
        # load config with game_interface.load_config()
        # sample from demoset num_demo_given
        # downsample with sampling_rate
        
    
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
    
    with torch.no_grad():  # Important for inference
        while t < CONFIGS['MAX_INFERENCE_ITER'] and not done:
            # Initialize with random noise
            noisy_actions = get_random_noisy_action(device)
            
            # DDIM denoising loop (reverse diffusion)
            timesteps = torch.linspace(num_diffusion_steps-1, 0, num_diffusion_steps, dtype=torch.long, device=device)
            
            for i, timestep in enumerate(timesteps):
                # Get noise prediction from agent
                predicted_noise, _ = agent(
                    curr_obs,
                    provided_demos,
                    noisy_actions
                )
                
                # DDIM denoising step
                if i < len(timesteps) - 1:  # Not the final step
                    noisy_actions = ddim_step(
                        agent.alpha_cumprod, 
                        alpha_cumprod_prev, 
                        noisy_actions, 
                        predicted_noise, 
                        timestep
                    )
                else:  # Final step - compute clean action
                    alpha_cumprod_t = agent.alpha_cumprod[timestep]
                    clean_actions = (
                        noisy_actions - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
                    ) / torch.sqrt(alpha_cumprod_t)
                    noisy_actions = clean_actions
            
            # Convert final denoised action to game action
            action_obj = action_tensor_to_obj(noisy_actions)
            
            # Take step in environment
            curr_obs = game_interface.step(action_obj)
            done = curr_obs.get('done', False)
            
            t += 1
            
            if t % 10 == 0:
                print(f"Step {t}, Done: {done}")
    
    print(f"Evaluation completed after {t} steps")