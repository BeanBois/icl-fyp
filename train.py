# import graph 
from utils.graph import LocalGraph, DemoGraph, ActionGraph, ContextGraph, make_localgraph


# import game
from tasks.twoD.game import GameInterface, PseudoGame, GameMode, PseudoGameMode

# import agent
from agent.agent import InstantPolicyAgent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt




# Generates a single X for training. 
#  X contains:
#       context : list of (obs, action)
#       clean_actions : the actions taken in the Nth demo
#       curr_obs : curr observation of the Nth demo
class PseudoDemoGenerator:

    def __init__(self, device, num_demos = 5, min_demo_length = 2, max_demo_length = 6, num_sampled_pointclouds = 20):
        # self.num_diffusion_steps = num_diffusion_steps
        self.num_demos = num_demos
        self.min_demo_length = min_demo_length
        self.max_demo_length = max_demo_length
        self.num_sampled_pc = num_sampled_pointclouds
        self.device = device
        # self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        # self.alpha_schedule = 1-self.beta_schedule
        # self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)

        self.translation_scale = 500
        self.max_translation = 100  # 100 pixels
        self.max_rotation = np.pi / 9  # 20 degrees


    def get_sample(self, mode):
        context = self._get_context(mode)
        # dont actaully need label HAHAHAH
        curr_obs, clean_actions, fututre_obs = self._get_ground_truth(mode)
        clean_actions = torch.tensor(clean_actions, device = self.device)
        return curr_obs, context, clean_actions, fututre_obs

    def _run_game(self, mode):
        max_retries = 5
        for attempt in range(max_retries):
            try: 
                pseudo_demo = PseudoGame(mode=mode, max_num_sampled_waypoints=self.max_demo_length, min_num_sampled_waypoints=self.min_demo_length, num_sampled_point_clouds=self.num_sampled_pc)
                pseudo_demo.run()
                return pseudo_demo
            except Exception as e:
                print(f'Attempted {attempt + 1} failed : {e}')
                if attempt == max_retries-1:
                    raise 
                continue

    def _get_context(self, mode):
        context = []
        for _ in range(self.num_demos - 1):
            pseudo_demo = self._run_game(mode)
            observations = pseudo_demo.observations 
            context.append(observations)
        return context
            
    def _get_ground_truth(self,mode):
        pseudo_demo = self._run_game(mode)
        true_obs = pseudo_demo.observations
        actions = torch.tensor(pseudo_demo.get_actions(), dtype=torch.float, device = self.device)       
        # true_actions = pseudo_demo.get_actions() 
        # action_dim = (len(true_actions), len(true_actions[0]))
        # actions = torch.zeros(size = action_dim, device=self.device)

        # for i, action in enumerate(true_actions):
        #     actions[i, ...] = torch.tensor(action, dtype=torch.float, device = self.device)

        return true_obs[0], actions, true_obs[1:]
    # def _get_noisy_actions(self,actions : torch.Tensor, timesteps : torch.Tensor):

    #     # Separate SE(2) transformations and binary state
    #     se2_actions = actions[..., :3]  # Translation (2) + rotation (1)
    #     binary_actions = actions[..., 3:4]  # Binary agent state
        
    #     # Project SE(2) to se(2) tangent space for noise addition
    #     se2_tangent = self._se2_to_tangent(se2_actions)
        
    #     # Normalize to [-1, 1] range
    #     se2_normalized = self._normalize_se2(se2_tangent)
        
    #     # Sample noise
    #     se2_noise = torch.randn_like(se2_normalized)
    #     binary_noise = torch.randn_like(binary_actions)
        
    #     # Add noise according to schedule
    #     alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1)
    #     # alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1, 1)
        
    #     noisy_se2 = torch.sqrt(alpha_cumprod_t) * se2_normalized + torch.sqrt(1 - alpha_cumprod_t) * se2_noise
    #     noisy_binary = torch.sqrt(alpha_cumprod_t) * binary_actions + torch.sqrt((1 - alpha_cumprod_t[..., 0]).unsqueeze(-1)) * binary_noise
        
    #     # Convert back to SE(2) for graph construction
    #     noisy_se2_unnorm = self._unnormalize_se2(noisy_se2)
    #     noisy_se2_actions = self._tangent_to_se2(noisy_se2_unnorm)
        
    #     noisy_actions = torch.cat([noisy_se2_actions, noisy_binary], dim=-1)
    #     noise = torch.cat([se2_noise, binary_noise], dim=-1)
    #     return noisy_actions, noise.float()
    
    # def _se2_to_tangent(self, se2_actions):
    #     """Convert SE(2) [x, y, theta] to se(2) tangent space"""
    #     # For SE(2), tangent space is just [x, y, theta] since it's already linear
    #     return se2_actions

    # def _tangent_to_se2(self, tangent):
    #     """Convert se(2) tangent space back to SE(2)"""
    #     return tangent

    # def _normalize_se2(self, se2_tangent):
    #     """Normalize SE(2) tangent vectors to [-1, 1]"""
    #     # You'll need to define appropriate normalization ranges
    #     # Example: normalize translations and rotation separately
    #     translation = se2_tangent[..., :2]  # [x, y]
    #     rotation = se2_tangent[..., 2:3]    # [theta]
        
    #     # Normalize translation (adjust ranges as needed)
    #     norm_translation = translation / self.translation_scale
    #     # Normalize rotation to [-1, 1] from [-π, π]
    #     norm_rotation = rotation / torch.pi
        
    #     return torch.cat([norm_translation, norm_rotation], dim=-1)

    # def _unnormalize_se2(self, normalized_se2):
    #     """Unnormalize SE(2) from [-1, 1] back to original ranges"""
    #     translation = normalized_se2[..., :2] * self.translation_scale
    #     rotation = normalized_se2[..., 2:3] * torch.pi
        
    #     return torch.cat([translation, rotation], dim=-1)

    

class Trainer:

    def __init__(self, agent, num_demos_for_context, min_demo_length = 2, max_demo_length = 6, num_sampled_pointclouds = 20, device='cuda'):
        self.agent = agent
        self.device = device 
        self.data_generator = PseudoDemoGenerator(
                num_demos = num_demos_for_context + 1,  
                max_demo_length = max_demo_length, 
                min_demo_length=min_demo_length,
                num_sampled_pointclouds=num_sampled_pointclouds,
                device=self.device
                )
  
        self.modes = [mode for mode in PseudoGameMode if mode != PseudoGameMode.RANDOM]
        # Optimizer (from appendix: AdamW with 1e-5 learning rate)
        self.optimizer = optim.AdamW(agent.parameters(), lr=1e-5)

        

    def train_step(self):
        mode = self.modes[random.randint(0,len(self.modes)-1)]
        curr_obs, context, clean_actions = self.data_generator.get_sample(mode) # clean action is a tensor

        predicted_noise, actual_noise = self.agent(
            curr_obs,
            context,
            clean_actions
        )


        # then MSE lost for 
        loss = nn.MSELoss()(predicted_noise, actual_noise)
        # loss = nn.MSELoss()(predictions, noise)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    
    def train_epoch(self, num_steps_per_epoch = 100):
        total_loss = 0.0
        for step in range(num_steps_per_epoch):
            # Train step
            loss = self.train_step()
            total_loss += loss
            
            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss:.6f}")
        
        return total_loss / num_steps_per_epoch
    
    def full_training(self, num_steps_per_epoch=200, num_epochs=50, 
                      save_model = True, save_path = "instant_policy.pth"):
        avg_losses = []
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(num_steps_per_epoch)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
            # Learning rate cooldown in final 50k steps
            if epoch > num_epochs - 4:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.99
            avg_losses.append(avg_loss)
        self.plot_losses(avg_losses,num_steps_per_epoch)
        if save_model:
            torch.save(self.agent, save_path)
            torch.save(self.alpha_schedule, f'{save_path[:5]}-alpha_schedule.pth')


    def plot_losses(self, losses,num_steps_per_epoch):
        plt.figure()
        plt.plot(losses)
        plt.title(f'avegrage losses from each epoch ({num_steps_per_epoch} steps)')
        plt.show()

            



# Usage example:
if __name__ == "__main__":
    from configs import CONFIGS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize agent
    agent = InstantPolicyAgent(
                device=device, 
                num_diffusion_steps=CONFIGS['NUM_DIFFUSION_STEPS'],
                num_agent_nodes=CONFIGS['NUM_AGENT_NODES'],
                pred_horizon=CONFIGS['PRED_HORIZON'],
                hidden_dim=CONFIGS['HIDDEN_DIM'],
                node_embd_dim=CONFIGS['NODE_EMB_DIM'],
                edge_embd_dim=CONFIGS['EDGE_EMB_DIM'],
                agent_state_embd_dim=CONFIGS['AGENT_STATE_EMB_DIM'],
                edge_pos_dim=CONFIGS['EDGE_POS_DIM']
                )
    
    
    # Initialize trainer
    trainer = Trainer(agent,
                      device=device, 
                      num_demos_for_context=CONFIGS['NUM_DEMO_GIVEN'],
                      max_demo_length=CONFIGS['DEMO_MAX_LENGTH'],
                      min_demo_length=CONFIGS['DEMO_MIN_LENGTH'],
                      num_sampled_pointclouds=CONFIGS['NUM_SAMPLED_POINTCLOUDS'],
                      )
    
    
    # Train the model
    trainer.full_training(
        save_path=CONFIGS['MODEL_FILE_PATH'],
        save_model=CONFIGS['SAVE_MODEL'],
        num_steps_per_epoch=CONFIGS['NUM_STEPS_PER_EPOCH'],
        num_epochs=CONFIGS['NUM_EPOCHS']
    )



