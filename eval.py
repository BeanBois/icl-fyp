import torch 
import numpy as np
from tasks.twoD.game import GameInterface, GameMode, Action, PlayerState





# 
def get_random_noisy_action(max_x, max_y, max_rot, state_change_odds):
    # paper get random noisy action from normal distribution
    actions = torch.rand(3)
    actions[0] *= max_x
    actions[1] *= max_y
    actions[2] *= max_rot
    roll = torch.multinomial(state_change_odds, num_samples=1)
    actions = torch.cat([actions,roll])
    return actions


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

if __name__ == "__main__":
    from configs import CONFIGS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = torch.load(CONFIGS['MODEL_FILE_PATH'])
    alpha_schedule = torch.load(f'{CONFIGS['MODEL_FILE_PATH'][:5]}-alpha_schedule.pth')

    agent.eval()

    # collect demos 
    _max_translation = 1000
    _max_x = 800
    _max_y = 600 
    _max_rot = 20
    _state_change_odds = torch.tensor([0.5,0.5]) 


    game_interface = GameInterface(
        mode=GameMode.DEMO_MODE,
        num_sampled_points=CONFIGS['NUM_SAMPLED_POINTCLOUDS']
    )
    num_demos_given = CONFIGS['NUM_DEMO_GIVEN']
    demo_length = CONFIGS['DEMO_LENGTH']
    

    provided_demos = []

    for n in range(num_demos_given):
        demo = [game_interface.start_game()]
        while game_interface.running:
            demo.append(game_interface.step())
        
        # process demo to only take out 
        processed_demos = []
        for i in range(0, len(demo), step = len(demo) // num_demos_given):
            processed_demos.append(demo[i])

        game_interface.reset()
        provided_demos.append(provided_demos)

    # then evaluate
    num_diffusion_steps = CONFIGS['NUM_DIFFUSION_STEPS']
    game_interface.change_mode(GameMode.AGENT_MODE)
    _t = 0
    curr_obs = game_interface.start_game()

    alpha_cumprod = torch.cumprod(alpha_schedule, dim=0)
    alpha_cumprod_prev = torch.cat([
            torch.tensor([1.0]), 
            alpha_cumprod[:-1]
    ])
    done = False

    while _t < CONFIGS['MAX_INFERENCE_ITER'] and not done:
        random_noisy_action = get_random_noisy_action(_max_x,_max_y, _max_rot, _state_change_odds)
        for k in range(num_diffusion_steps):
            predicted_noise = agent(
                curr_obs,
                provided_demos,
                random_noisy_action
            )
            action = ddim_step(alpha_cumprod, alpha_cumprod_prev, random_noisy_action, predicted_noise,k)
        
        # now create action_obj from aciton

        action_obj = Action(
            forward_movement= (action[0] ** 2 + action[1] ** 2) ** 0.5,
            rotation= np.rad2deg(action[3]),
            state_change = PlayerState.EATING if action[4] == 1 else PlayerState.NOT_EATING
        )

        curr_obs = game_interface.step(action_obj)
        done = curr_obs['done']
        _t += 1

    


    






    