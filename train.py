# import game
from configs import SCREEN_HEIGHT, SCREEN_WIDTH, PseudoGame, action_mode

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




class PseudoDemoGenerator:

    def __init__(self, device, num_demos=5, min_num_waypoints=2, max_num_waypoints=6, 
                 num_threads=4, demo_length = 10):
        self.num_demos = num_demos
        self.min_num_waypoints = min_num_waypoints
        self.max_num_waypoints = max_num_waypoints
        self.device = device
        self.agent_key_points = PseudoGame.agent_keypoints
        self.translation_scale = 500
        self.demo_length = demo_length
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
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all sample generation tasks

            futures = [executor.submit(self._generate_single_sample) for _ in range(batch_size)]
            
            # Collect results as they complete
            curr_obs_batch = []
            context_batch = []
            clean_actions_list = []

            
            for future in as_completed(futures):
                curr_obs, context, clean_actions = future.result()
                curr_obs_batch.append(curr_obs)
                context_batch.append(context)
                clean_actions_list.append(clean_actions)

        
        # Stack clean actions into a single tensor [batch_size, pred_horizon, 4]
        clean_actions_batch = torch.stack(clean_actions_list, dim=0)

        
        # Store agent keypoints from the last generated sample (they should all be the same)
        if hasattr(self._thread_local, 'agent_key_points') and self._thread_local.agent_key_points is not None:
            self.agent_key_points = self._thread_local.agent_key_points
        
        return curr_obs_batch, context_batch, clean_actions_batch

    def _generate_single_sample(self) -> Tuple[dict, List, torch.Tensor]:
        """Generate a single training sample (thread-safe)"""
        biased = np.random.rand() < self.biased_odds
        augmented = self.augmented # for now 
        pseudo_game = self._make_game(biased, augmented)
        context = self._get_context(pseudo_game)   
        curr_obs, clean_actions = self._get_ground_truth(pseudo_game)
        return curr_obs, context, clean_actions

    # TODO : FIX
    def get_agent_keypoints(self):
            
        agent_keypoints = torch.zeros((len(self.agent_key_points), 2), device=self.device)
        agent_keypoints[0] = torch.tensor(self.agent_key_points['front'], device=self.device)
        agent_keypoints[1] = torch.tensor(self.agent_key_points['back-left'], device=self.device)
        agent_keypoints[2] = torch.tensor(self.agent_key_points['back-right'], device=self.device)
        agent_keypoints[3] = torch.tensor(self.agent_key_points['center'], device=self.device)
        return agent_keypoints
    
    def _make_game(self, biased,augmented):
        player_starting_pos =(random.randint(0,SCREEN_WIDTH), random.randint(0,SCREEN_HEIGHT))
        return PseudoGame(
                    player_starting_pos=player_starting_pos,
                    max_num_sampled_waypoints=self.max_num_waypoints, 
                    min_num_sampled_waypoints=self.min_num_waypoints, 
                    biased=biased, 
                    augmented=augmented
                )

    def _run_game(self, pseudo_demo):
        max_retries = 1000
        player_starting_pos =(random.randint(0,SCREEN_WIDTH), random.randint(0,SCREEN_HEIGHT))
        for attempt in range(max_retries):
            try: 
                # first reset 
                pseudo_demo.reset_game(shuffle=True) # config stays, but game resets (player, obj change positions)

                pseudo_demo.run()
                return pseudo_demo
            except Exception as e:
                if attempt == max_retries-1:
                    raise 
                continue

    def _get_context(self, pseudo_game):
        context = []
        for _ in range(self.num_demos - 1):
            pseudo_demo = self._run_game(pseudo_game)
            observations = pseudo_demo.observations
            sample_rate = len(observations) // self.demo_length
            sampled_obs = observations[::sample_rate][:self.demo_length]
            context.append(sampled_obs)
        return context
            
    def _get_ground_truth(self, pseudo_game):
        pseudo_game.set_augmented(np.random.rand() > 0.5) 
        pseudo_demo = self._run_game(pseudo_game)
        pd_actions = pseudo_demo.get_actions(mode='se2')

        se2_actions = np.array([action[0].flatten() for action in pd_actions]) # n x 9
        state_actions = np.array([action[1] for action in pd_actions]) # n x 1
        state_actions = state_actions.reshape(-1,1)
        actions = np.concatenate([se2_actions, state_actions], axis=1)
        actions = torch.tensor(
            actions, 
            dtype=torch.float, 
            device=self.device
        )          
        temp = actions.shape
        actions = self._accumulate_actions(actions)
        sample_rate = actions.shape[0] // self.demo_length
        assert temp == actions.shape
        actions = actions[::sample_rate][:self.demo_length]
        true_obs = pseudo_demo.observations[::sample_rate][:self.demo_length]


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

    def _process_demos(self):
        return    

class Trainer: 

    def __init__(self, agent, num_demos_for_context, num_agent_nodes = 4,pred_horizon = 8, min_num_waypoints = 2, max_num_waypoints = 6, demo_length=10, batch_size = 10,device='cuda'):
        self.agent = agent
        self.device = device 
        self.data_generator = PseudoDemoGenerator(
                num_demos = num_demos_for_context + 1,  
                max_num_waypoints = max_num_waypoints, 
                min_num_waypoints=min_num_waypoints,
                demo_length = demo_length,
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
            predicted_per_node_noise, action_noise_flat = self.agent(curr_obs, context, clean_actions, action_mode) 

            actual_pn_denoising_dir = self.get_pnn_from_noise(agent_keypoints, action_noise_flat)
            assert predicted_per_node_noise.shape == actual_pn_denoising_dir.shape # (num_agent_nodes, 3, 1)

            # Direct MSE loss in action space
            loss = nn.MSELoss()(predicted_per_node_noise, actual_pn_denoising_dir)
            scaled_loss = loss / self.batch_size
            scaled_loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / self.batch_size
    
    # TODO : test for this
    def get_pnn_from_noise(eslf,agent_keypoints, action_noise):
        """
        Calculate per-node denoising directions from noisy actions
        
        Args:
            agent_keypoints: (4, 2) - current gripper keypoints [x, y]
            action_noise: (L, 10) - noisy cumulative actions [SE(2) + state]
        
        Returns:
            denoising_directions: (L, 4, 5) - per node denoising directions
            where 5 = [delta_x, delta_y, delta_rot_x, delta_rot_y, delta_gripper]
        """
        L = action_noise.shape[0]
        
        # Original keypoints repeated for each action
        original_keypoints = agent_keypoints.unsqueeze(0).repeat(L, 1, 1)  # (L, 4, 2)
        
        # Extract SE(2) transformations (first 9 dims)
        se2_noise = action_noise[:, :9].view(L, 3, 3)  # (L, 3, 3)
        state_noise = action_noise[:, 9:10]  # (L, 1)
        
        # Convert to homogeneous coordinates
        ones = torch.ones(L, 4, 1, device=agent_keypoints.device, dtype=agent_keypoints.dtype)
        keypoints_homo = torch.cat([original_keypoints, ones], dim=-1)  # (L, 4, 3)
        
        # Apply SE(2) transformations
        transformed_keypoints_homo = torch.bmm(keypoints_homo, se2_noise.transpose(-2, -1))
        transformed_keypoints = transformed_keypoints_homo[:, :, :2]  # (L, 4, 2)
        
        # Calculate translation shifts (actual displacement)
        translation_shifts = transformed_keypoints - original_keypoints  # (L, 4, 2)
        
        # Calculate rotation components (for denoising)
        # Extract rotation part of SE(2) matrix
        rotation_matrices = se2_noise[:, :2, :2]  # (L, 2, 2)
        
        # Calculate rotation-induced shifts for each keypoint
        centered_keypoints = original_keypoints - original_keypoints.mean(dim=1, keepdim=True)
        rotated_centered = torch.bmm(centered_keypoints, rotation_matrices.transpose(-2, -1))
        rotation_shifts = rotated_centered - centered_keypoints  # (L, 4, 2)
        
        # Repeat gripper state for each keypoint
        gripper_shifts = state_noise.unsqueeze(1).repeat(1, 4, 1)  # (L, 4, 1)
        
        # Combine all denoising directions: [trans_x, trans_y, rot_x, rot_y, gripper]
        denoising_directions = torch.cat([
            translation_shifts,      # (L, 4, 2) - [delta_x, delta_y]
            rotation_shifts,         # (L, 4, 2) - [delta_rot_x, delta_rot_y] 
            gripper_shifts          # (L, 4, 1) - [delta_gripper]
        ], dim=-1)  # (L, 4, 5)
        
        return denoising_directions

    
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
    print(f'training agent: {_type}, version: {version}, action mode : {action_mode}')
    # train geometry encoder
    geometry_encoder_filename = f'geometry_encoder_2d_v{geo_version}.pth'
    node_embd_dim = CONFIGS['NUM_ATT_HEADS'] * CONFIGS['HEAD_DIM']
    grouping_radius = CONFIGS['GROUPING_RADIUS']
    num_sampled_pc = CONFIGS['NUM_SAMPLED_POINTCLOUDS']
    if CONFIGS['TRAIN_GEO_ENCODER']:
        full_train(num_sampled_pc, node_embd_dim, device, grouping_radius, filename=geometry_encoder_filename, num_epochs= CONFIGS['GEO_NUM_EPOCHS'], num_samples= CONFIGS['GEO_BATCH_SIZE'])

    model = GeometryEncoder2D(num_centers=num_sampled_pc,radius=grouping_radius, node_embd_dim=node_embd_dim, device=device).to(device)
    geometry_encoder = initialise_geometry_encoder(model, geometry_encoder_filename,device=device)

    # Initialize agent
    agent = InstantPolicyAgent(
                device=device, 
                geometry_encoder=geometry_encoder,
                max_translation=CONFIGS['TRAINING_MAX_TRANSLATION'],
                max_rotation=CONFIGS['MAX_ROTATION_DEG'],
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
                    max_num_waypoints=CONFIGS['MAX_NUM_WAYPOINTS'],
                    min_num_waypoints=CONFIGS['MIN_NUM_WAYPOINTS'],
                    demo_length=CONFIGS['DEMO_MAX_LENGTH'],
                    batch_size=CONFIGS['BATCH_SIZE']
                    )
    
    # Train the model
    trainer.full_training(
        save_path=CONFIGS['MODEL_FILE_PATH'],
        save_model=CONFIGS['SAVE_MODEL'],
        num_steps_per_epoch=CONFIGS['NUM_STEPS_PER_EPOCH'],
        num_epochs=CONFIGS['NUM_EPOCHS']
    )



