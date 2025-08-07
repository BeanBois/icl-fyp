# import game
from configs import SCREEN_HEIGHT, SCREEN_WIDTH, PseudoGame

# import agent
from agent_files import InstantPolicyAgent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple, Dict

# move these constants to CONFIG FILES 



# Generates a single X for training. 
#  X contains:
#       context : list of (obs, action)
#       clean_actions : the actions taken in the Nth demo
#       curr_obs : curr observation of the Nth demo

# think about diff demo types, biaseed, augmeneted ect
class PseudoDemoGenerator:

    def __init__(self, device, num_demos=5, min_num_waypoints=2, max_num_waypoints=6, 
                 sample_rate=5, num_threads=4):
        self.num_demos = num_demos
        self.min_num_waypoints = min_num_waypoints
        self.max_num_waypoints = max_num_waypoints
        self.device = device
        self.sample_rate = sample_rate
        self.agent_key_points = None
        self.translation_scale = 500
        # change this to take in argument instead
        self.max_translation = CONFIGS['TRAINING_MAX_TRANSLATION']
        self.max_rotation = np.deg2rad(CONFIGS['MAX_ROTATION_DEG'])
        self.player_speed = 5 
        self.player_rot_speed = 5
        self.num_threads = num_threads
        self.biased_odds = 0.1
        self.augmented = True
        
        # Thread-local storage for agent keypoints
        self._thread_local = threading.local()

    def get_batch_samples(self, batch_size: int) -> Tuple[torch.Tensor, List, torch.Tensor]:
        """
        Generate a batch of samples in parallel
        Returns:
            curr_obs_batch: List of batch_size current observations
            context_batch: List of batch_size contexts (each context is a list of demos)
            clean_actions_batch: Tensor of shape [batch_size, pred_horizon, 4]
        """
        # Use ThreadPoolExecutor to generate samples in parallel
        # TODO : to add random, biased and augmented here 
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all sample generation tasks
            futures = [executor.submit(self._generate_single_sample) for _ in range(batch_size)]
            
            # Collect results as they complete
            curr_obs_batch = []
            context_batch = []
            clean_actions_list = []
            prev_stacked_actions_list = []

            
            for future in as_completed(futures):
                curr_obs, context, clean_actions, prev_stacked_actions = future.result()
                curr_obs_batch.append(curr_obs)
                context_batch.append(context)
                clean_actions_list.append(clean_actions)
                prev_stacked_actions_list.append(prev_stacked_actions)

        
        # Stack clean actions into a single tensor [batch_size, pred_horizon, 4]
        clean_actions_batch = torch.stack(clean_actions_list, dim=0)
        prev_stacked_actions_batch = torch.stack(prev_stacked_actions_list, dim=0)

        
        # Store agent keypoints from the last generated sample (they should all be the same)
        if hasattr(self._thread_local, 'agent_key_points') and self._thread_local.agent_key_points is not None:
            self.agent_key_points = self._thread_local.agent_key_points
        
        return curr_obs_batch, context_batch, clean_actions_batch, prev_stacked_actions_batch

    def _generate_single_sample(self) -> Tuple[dict, List, torch.Tensor]:
        """Generate a single training sample (thread-safe)"""
        context = self._get_context()   
        curr_obs, clean_actions = self._get_ground_truth()
        return curr_obs, context, clean_actions

    def get_agent_keypoints(self):
        if self.agent_key_points is None:
            return torch.zeros((4, 2), device=self.device)
            
        agent_keypoints = torch.zeros((len(self.agent_key_points), 2), device=self.device)
        agent_keypoints[0] = torch.tensor(self.agent_key_points['front'], device=self.device)
        agent_keypoints[1] = torch.tensor(self.agent_key_points['back-left'], device=self.device)
        agent_keypoints[2] = torch.tensor(self.agent_key_points['back-right'], device=self.device)
        agent_keypoints[3] = torch.tensor(self.agent_key_points['center'], device=self.device)
        return agent_keypoints
    
    def _run_game(self, biased, augmented):
        max_retries = 1000
        player_starting_pos =(random.randint(0,SCREEN_WIDTH), random.randint(0,SCREEN_HEIGHT))
        for attempt in range(max_retries):
            try: 
                pseudo_demo = PseudoGame(
                    player_starting_pos=player_starting_pos,
                    max_num_sampled_waypoints=self.max_num_waypoints, 
                    min_num_sampled_waypoints=self.min_num_waypoints, 
                    biased=biased, 
                    augmented=augmented
                )
                # Store in thread-local storage
                if not hasattr(self._thread_local, 'agent_key_points'):
                    self._thread_local.agent_key_points = pseudo_demo.get_player_keypoints()
                pseudo_demo.run()
                return pseudo_demo
            except Exception as e:
                if attempt == max_retries-1:
                    raise 
                continue

    def _get_context(self):
        context = []
        for _ in range(self.num_demos - 1):
            biased = random.random() < self.biased_odds
            augmented = self.augmented
            pseudo_demo = self._run_game(biased=biased, augmented=augmented)
            observations = pseudo_demo.observations
            sampled_obs = observations[::self.sample_rate]
            context.append(sampled_obs)
        return context
            
    def _get_ground_truth(self):
        pseudo_demo = self._run_game(biased=True, augmented=False)
        true_obs = pseudo_demo.observations[::self.sample_rate]
        pd_actions = pseudo_demo.get_actions(mode='se2')

        se2_actions = np.array([action[0] for action in pd_actions]) # n x 3 x 3
        state_actions = np.array([action[1] for action in pd_actions]) # n x 1

        se2_actions = np.array(se2_actions).reshape(-1,9)
        actions = np.concat([se2_actions, state_actions])

        actions = torch.tensor(
            actions, 
            dtype=torch.float, 
            device=self.device
        )          

        actions = self._accumulate_actions(actions)
        actions = actions[::self.sample_rate]
        return true_obs[0], actions
    
    def _accumulate_actions(self, actions):
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
    

    


class Trainer: 

    def __init__(self, agent, num_demos_for_context, num_agent_nodes = 4,pred_horizon = 8, min_demo_length = 2, max_demo_length = 6, batch_size = 10,device='cuda'):
        self.agent = agent
        self.device = device 
        self.data_generator = PseudoDemoGenerator(
                num_demos = num_demos_for_context + 1,  
                max_num_waypoints = max_demo_length, 
                min_num_waypoints=min_demo_length,
                device=self.device
                )
        self.agent_keypoint = None
        self.batch_size = batch_size
        self.pred_horizon = pred_horizon
        self.num_agent_nodes = num_agent_nodes
        # Optimizer (from appendix: AdamW with 1e-5 learning rate)
        self.optimizer = optim.AdamW(agent.parameters(), lr=1e-5)
      
    def train_step(self):
        self.optimizer.zero_grad()
        total_loss = 0.0
        curr_obs_batch, context_batch, clean_actions_batch = self.data_generator.get_batch_samples(self.batch_size)
        agent_keypoints = self.data_generator.get_agent_keypoints()
        
        for i in range(self.batch_size):
            curr_obs, context, clean_actions = curr_obs_batch[i], context_batch[i], clean_actions_batch[i]

            predicted_per_node_noise, action_noise_flat = self.agent(curr_obs, context, clean_actions) # N x 

            # we will fix later HAHHAHA fuck this bull shit i should kill myself!
            actual_pn_denoising_dir = self.get_pnn_from_noise(agent_keypoints, action_noise_flat)

            assert predicted_per_node_noise.shape == actual_pn_denoising_dir.shape # (N, 3, 1)

            # Direct MSE loss in action space
            loss = nn.MSELoss()(predicted_per_node_noise, actual_pn_denoising_dir)
            scaled_loss = loss / self.batch_size
            scaled_loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / self.batch_size
    
    def get_pnn_from_noise(self, agent_keypoints, action_noise_flat):
        # Reshape SE(2) matrices from flattened format
        state_action_noise = action_noise_flat[...,-1]


        se2_matrices = action_noise_flat[...,:-1].view(self.batch_size, self.pred_horizon, 3, 3)
        # Convert keypoints to homogeneous coordinates [x, y, 1]
        ones = torch.ones(self.batch_size, self.pred_horizon, self.num_agent_nodes, 1, 
                        device=agent_keypoints.device, dtype=agent_keypoints.dtype)
        keypoints_homo = torch.cat([agent_keypoints, ones], dim=-1)  # (B, T, N, 3)
        
        # Apply SE(2) transformation: p_new = T @ p_old
        # keypoints_homo: (B, T, N, 3) -> (B, T, N, 3, 1) for batch matrix multiply
        keypoints_homo = keypoints_homo.unsqueeze(-1)
        
        # se2_matrices: (B, T, 3, 3) -> (B, T, 1, 3, 3) to broadcast over keypoints
        se2_matrices = se2_matrices.unsqueeze(2)
        
        # Matrix multiplication: (B, T, N, 3, 3) @ (B, T, N, 3, 1) -> (B, T, N, 3, 1)
        transformed_homo = torch.matmul(se2_matrices, keypoints_homo).squeeze(-1)
        
        # Extract x, y coordinates (ignore homogeneous coordinate)
        noisy_keypoints = transformed_homo[:, :, :, :2]

        
        return noisy_keypoints

    
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
        if save_model:
            torch.save(self.agent, save_path)
        self.plot_losses(avg_losses,num_steps_per_epoch)

    def plot_losses(self, losses,num_steps_per_epoch, display = False):
        fig = plt.figure()
        plt.plot(losses)
        plt.title(f'avegrage losses from each epoch ({num_steps_per_epoch} steps)')
        if display:
            plt.show()
        plt.savefig(CONFIGS['FIG_FILENAME'])

            



# Usage example:
if __name__ == "__main__":
    from configs import CONFIGS, version, _type, geo_version
    from agent_files import GeometryEncoder2D, full_train, initialise_geometry_encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # print statement
    print(f'training agent: {_type}, version: {version}')
    # train geometry encoder
    geometry_encoder_filename = f'geometry_encoder_2d_v{geo_version}.pth'
    node_embd_dim = CONFIGS['NUM_ATT_HEADS'] * CONFIGS['HEAD_DIM']
    grouping_radius = CONFIGS['GROUPING_RADIUS']
    full_train(node_embd_dim, device, grouping_radius, filename=geometry_encoder_filename, num_epochs= CONFIGS['GEO_NUM_EPOCHS'], num_samples= CONFIGS['GEO_BATCH_SIZE'])
    model = GeometryEncoder2D(radius=grouping_radius, node_embd_dim=node_embd_dim, device=device).to(device)
    geometry_encoder = initialise_geometry_encoder(model, geometry_encoder_filename,device=device)

    # Initialize agent
    agent = InstantPolicyAgent(
                device=device, 
                geometry_encoder=geometry_encoder,
                max_translation=CONFIGS['TRAINING_MAX_TRANSLATION'],
                num_diffusion_steps=CONFIGS['NUM_DIFFUSION_STEPS'],
                num_agent_nodes=CONFIGS['NUM_AGENT_NODES'],
                pred_horizon=CONFIGS['PRED_HORIZON'],
                num_att_heads=CONFIGS['NUM_ATT_HEADS'],
                head_dim=CONFIGS['HEAD_DIM'],
                agent_state_embd_dim=CONFIGS['AGENT_STATE_EMB_DIM'],
                edge_pos_dim=CONFIGS['EDGE_POS_DIM']
                )
    
    
    # Initialize trainer
    trainer = Trainer(agent,
                    device=device, 
                    num_agent_nodes=CONFIGS['NUM_AGENT_NODES'],
                    pred_horizon=CONFIGS['PRED_HORIZON'],
                    num_demos_for_context=CONFIGS['NUM_DEMO_GIVEN'],
                    max_demo_length=CONFIGS['DEMO_MAX_LENGTH'],
                    min_demo_length=CONFIGS['DEMO_MIN_LENGTH'],
                    batch_size=CONFIGS['BATCH_SIZE']
                    )
    
    # Train the model
    trainer.full_training(
        save_path=CONFIGS['MODEL_FILE_PATH'],
        save_model=CONFIGS['SAVE_MODEL'],
        num_steps_per_epoch=CONFIGS['NUM_STEPS_PER_EPOCH'],
        num_epochs=CONFIGS['NUM_EPOCHS']
    )



