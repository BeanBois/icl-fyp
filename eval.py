import torch 

from tasks.twoD.game import GameInterface, GameMode





def get_random_noisy_action():
    # paper get random noisy action from normal distribution
    
    pass


if __name__ == "__main__":
    from configs import CONFIGS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = torch.load(CONFIGS['MODEL_FILE_PATH'])

    agent.eval()

    # collect demos 
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
    game_interface.change_mode(GameMode.AGENT_MODE)
    _t = 0
    curr_obs = game_interface.start_game()
    while _t < CONFIGS['MAX_INFERENCE_ITER']:
        random_noisy_action = get_random_noisy_action()
        action = agent(
            curr_obs,
            provided_demos,
            random_noisy_action
        )



    






    