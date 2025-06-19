# import graph 
from utils.graph import LocalGraph, DemoGraph, ActionGraph, ContextGraph, make_localgraph


# import game
from tasks.twoD.game import GameInterface, PseudoGame, GameMode

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

 

# training steps: training with pseudo demos
    # First we generate pseudo-demos
        # Then convert pseudo-demos into graph representation (DemoGraph)
    # Then we move into the forward process 
        # Here we construct a noise-altered graph by adding noise to robot actions according to eq:
            # q(Gk |Gk−1) = G(Gal (N(ak; 1−βkak−1,βkI),Gc)), k= 1,...,K
                #  N is normal dist 
                # b_k is var scheduler 
                # K is total num of diffusion steps
    # Then we move onto the reverse process 
        # Here we use a parametrised model to reverse the diffusion process

# Generates a pseudo demo 
# then it converts this pseudo-demo into ground truth form 
# then it does the forward process 
class PseudoDemoGenerator:

    def __init__(self):
        self.num_diffusion_steps = 1000  # K in equations
        self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        self.alpha_schedule = np.sqrt(1-self.beta_schedule)
        self.final_action = None
        self.ground_truth = self._get_ground_truth()
        self.noisy_graph = self._get_noisy_graph()
        pass 


    def _linear_beta_schedule(self, beta_start, beta_end, timesteps):
        """Create linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def _get_ground_truth(self):
        pseudo_demo = PseudoGame()
        pseudo_demo.run()
        # first make demo 
        observations = pseudo_demo.observations[:-1]
        localgraphs = [make_localgraph(obs) for obs in observations]
        demo_graph = DemoGraph(localgraphs)

        curr_observation = pseudo_demo.observations[-2]
        curr_graph = make_localgraph(curr_observation)
        context_graph = ContextGraph(current_graph=curr_graph, demo_graphs=[demo_graph])

        final_action = pseudo_demo.final_action
        self.final_action = final_action
        final_action_graph = ActionGraph(context_graph=context_graph, action=final_action)

        # redo actions linking
        final_observation = pseudo_demo.observations[-1]
        ground_truth_graph = make_localgraph(final_observation)
        final_action_graph.predicted_graph = ground_truth_graph
        final_action_graph._connect_curr_graph_to_predicted_graph()

        return final_action_graph
    
    def _get_noisy_graph(self):

        context_graph = self.ground_truth.context_graph 

        # make noisy action now 
        # noisy process = N(a^k; sqrt(1-beta_k)a^k-1, beta_k * I)
        # action is 2d so
        noisy_action = self.final_action
        for i in range(self.num_diffusion_steps):
            noisy_action = np.random.normal(noisy_action * self.alpha_schedule[i], self.beta_schedule[i] * np.identity(2))

        noisy_graph = ActionGraph(context_graph=context_graph, action=noisy_action)

        return noisy_graph
    






class Trainer:

    def __init__(self, agent, device='cuda'):
        self.agent = agent
        self.device = device 
  
        # Diffusion parameters (from the paper)
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)
        
        # Optimizer (from appendix: AdamW with 1e-5 learning rate)
        self.optimizer = optim.AdamW(agent.parameters(), lr=1e-5)
        
        # Action normalization bounds (from appendix)
        self.max_translation = 0.01  # 1cm
        self.max_rotation = np.pi / 60  # 3 degrees

    









    def train(model):

        # 


        pass 



    def _get_pseudo_batch(batch_size = 20, num_iter = 50):
        batches = []
        for it in range(num_iter):
            pseudos = [PseudoGame() for _ in batch_size]
            [pseudo.run() for pseudo in pseudos]
            observations = [pseudo.observations for pseudo in pseudos]
            batches.append(observations)
        return batches

        
    def _get_demos(num_demos = 1):
        demos = []
        for _ in range(num_demos):
            observations = []
            game = GameInterface(mode = GameMode.DEMO_MODE)
            game.start_game()
            while game.running:
                obs = game.step()
                observations.append(obs)

            observations.pop(-1) # last element always None
            demos.append(observations)


        return demos 
        
    def _train(model, ):
        pass 



