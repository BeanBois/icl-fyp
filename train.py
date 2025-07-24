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
        self.agent_key_points = None
        # self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        # self.alpha_schedule = 1-self.beta_schedule
        # self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)

        self.translation_scale = 500
        self.max_translation = 100  # 100 pixels
        self.max_rotation = np.pi / 9  # 20 degrees


    def get_sample(self, mode):
        context = self._get_context(mode)   
        curr_obs, clean_actions, fututre_obs = self._get_ground_truth(mode)
        clean_actions = torch.tensor(clean_actions, device = self.device)
        return curr_obs, context, clean_actions

    def get_agent_keypoints(self):
        agent_keypoints = torch.zeros((len(self.agent_key_points), 2))
        agent_keypoints[0] = torch.tensor(self.agent_key_points['front'], device=self.device)
        agent_keypoints[1] = torch.tensor(self.agent_key_points['back-left'], device=self.device)
        agent_keypoints[2] = torch.tensor(self.agent_key_points['back-right'], device=self.device)
        agent_keypoints[3] = torch.tensor(self.agent_key_points['center'], device=self.device)
        return agent_keypoints
    
    def _run_game(self, mode):
        max_retries = 1000
        for attempt in range(max_retries):
            try: 
                pseudo_demo = PseudoGame(mode=mode, max_num_sampled_waypoints=self.max_demo_length, min_num_sampled_waypoints=self.min_demo_length, num_sampled_point_clouds=self.num_sampled_pc)
                self.agent_key_points = pseudo_demo.get_player_keypoints()
                pseudo_demo.run()
                return pseudo_demo
            except Exception as e:
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
        # need to change here too
        true_obs = pseudo_demo.observations
        actions = torch.tensor(pseudo_demo.get_actions(), dtype=torch.float, device = self.device)          
        actions = self._process_actions(actions) # need to cat action tgt  s.t it stacks
        return true_obs[0], actions, true_obs[1:]
    
    def _process_actions(self, actions):
        for i in range(len(actions) - 1):
            actions[i+1, :3] += actions[i, : 3] 
            actions[i+1, 2] = actions[i+1, 2] % 360 
        return actions 


    

class Trainer: 

    def __init__(self, agent, num_demos_for_context, min_demo_length = 2, max_demo_length = 6, num_sampled_pointclouds = 20, batch_size = 10,device='cuda'):
        self.agent = agent
        self.device = device 
        self.data_generator = PseudoDemoGenerator(
                num_demos = num_demos_for_context + 1,  
                max_demo_length = max_demo_length, 
                min_demo_length=min_demo_length,
                num_sampled_pointclouds=num_sampled_pointclouds,
                device=self.device
                )
        self.batch_size = batch_size
        self.modes = [mode for mode in PseudoGameMode if mode != PseudoGameMode.RANDOM]
        # Optimizer (from appendix: AdamW with 1e-5 learning rate)
        self.optimizer = optim.AdamW(agent.parameters(), lr=1e-5)

    def train_step(self):
        self.optimizer.zero_grad()
        total_loss = 0.0
        
        # Process each sample in the batch
        for i in range(self.batch_size):
            mode = self.modes[random.randint(0, len(self.modes)-1)]
            curr_obs, context, clean_actions = self.data_generator.get_sample(mode)
            
            predicted_noise, raw_noise = self.agent(
                curr_obs,
                context,
                clean_actions
            )
            
            agent_obj_keypoints = self.data_generator.get_agent_keypoints()
            actual_noise = self._get_position_noise(raw_noise, agent_obj_keypoints)
            
            # Compute loss for this sample
            loss = nn.MSELoss()(predicted_noise, actual_noise)
            
            # Scale loss by batch size and accumulate gradients
            scaled_loss = loss / self.batch_size
            scaled_loss.backward()
            
            total_loss += loss.item()
        
        # Single optimization step after accumulating all gradients
        self.optimizer.step()
        
        return total_loss / self.batch_size

    # def train_step(self):
    #     mode = self.modes[random.randint(0,len(self.modes)-1)]
    #     curr_obs, context, clean_actions = self.data_generator.get_sample(mode) # clean action is a tensor
        
    #     # actual noise should be a N x num-agent-nodes x 2 rep noise in positions of each agent node caused by actions 
    #     # to get actual noise from noisy actions: either
    #         # 1) treat noise as future-obs - predicted-action-graph-pos ( here then future obs need to be part of arg)
    #         # 2) extract noise using k.p, clean-actions and noisy actions (doable lmao, put this in agent code)
    #             # we have noise added to clean actions (noisy-actions - clean-actions) T_delta
    #             # with T_delta we can 
    #             # no we do this in train.py actually
    #             # with raw_noise being noise added to actions 
    #             # we process this raw_noise to produce noisy-positions with agent kp
    #             # taking center as reference point (so for this noise added is literally just 2d-space the T_delta)
    #             # then use other kps (wrt center) to calculate noise added to those 
        
    #     predicted_noise, raw_noise = self.agent(
    #         curr_obs,
    #         context,
    #         clean_actions
    #     )
    #     agent_obj_keypoints = self.data_generator.get_agent_keypoints() # add this in data_generator 

    #     actual_noise = self._get_position_noise(raw_noise, agent_obj_keypoints)

    #     # then MSE lost for 
    #     loss = nn.MSELoss()(predicted_noise, actual_noise)

    #     # backpropagation
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    #     return loss.item()
   
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

    # check this fn
    # raw-noise : N  x (x,y,theta-rad,state)
    # raw noise is actions (x_t, y_t, theta_deg) and state_noise  
    def _get_position_noise(self,raw_noise, agent_key_points):
        noise_size = (*raw_noise.shape[:2], 3) # N x num-agent-nodes x (x,y,state)
        actual_noise = torch.zeros(size = noise_size, device = self.device)
        # translation_noise = (raw_noise[:,0]**2 + raw_noise[:,1]**2)**0.5 # find torch op for this # this is p_delta
        translation_noise = raw_noise[:,:2]

        rotation_mat = torch.zeros((raw_noise.shape[0],2,2))
        theta_rad = torch.deg2rad(raw_noise[:,2])
        breakpoint()
        rotation_mat[:, 0,0] = torch.cos(theta_rad)
        rotation_mat[:, 1,1] = torch.cos(theta_rad)
        rotation_mat[:, 0,1] = -torch.sin(theta_rad)
        rotation_mat[:, 1,0] = torch.sin(theta_rad)
        # to get rot-noise, assuming kpp => (0,0) ... in (num-agent-nodesx2)
        rotation_noise = torch.einsum('ij,njk->nik', agent_key_points, rotation_mat)

        translation_noise_expanded = translation_noise.unsqueeze(1)  # Shape: (N, 1, 2)
        actual_noise = translation_noise_expanded + rotation_noise  # Shape: (N, 4, 2)
        

        state_noise = raw_noise[:, 3:4]
        state_noise_expanded = state_noise.unsqueeze(1).expand(-1, 4, -1)  # Shape: (N, 4, 1)
        full_actual_noise = torch.cat([actual_noise, state_noise_expanded], dim=-1)  # Shape: (N, 4, 3)
        return full_actual_noise
            



# Usage example:
if __name__ == "__main__":
    from configs import CONFIGS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Initialize agent
    agent = InstantPolicyAgent(
                device=device, 
                max_translation=CONFIGS['TRAINING_MAX_TRANSLATION'],
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
                      batch_size=CONFIGS['BATCH_SIZE']
                      )
    
    
    # Train the model
    trainer.full_training(
        save_path=CONFIGS['MODEL_FILE_PATH'],
        save_model=CONFIGS['SAVE_MODEL'],
        num_steps_per_epoch=CONFIGS['NUM_STEPS_PER_EPOCH'],
        num_epochs=CONFIGS['NUM_EPOCHS']
    )



