import torch
import numpy as np
import random
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tasks2d import LousyPacmanPseudoGame as PseudoGame 
from tasks2d import LousyPacmanTensorizedPseudoGame as TensorizedPseudoGame
from configs import PSEUDO_SCREEN_HEIGHT, PSEUDO_SCREEN_WIDTH

# TODO: 
# deal with demo length i think?
# also the padding is a problem 



# unused for now
class TensorizedPseudoDemoGenerator:
    """
    Tensorized version of PseudoDemoGenerator that maintains the same API
    but uses efficient batch processing under the hood.
    """
    
    def __init__(self, device, num_demos=5, min_num_waypoints=2, max_num_waypoints=6, 
                 num_threads=4, demo_length=10, batch_size=32):
        self.num_demos = num_demos
        self.min_num_waypoints = min_num_waypoints
        self.max_num_waypoints = max_num_waypoints
        self.device = device
        self.agent_key_points = PseudoGame.agent_keypoints
        self.translation_scale = 500
        self.demo_length = demo_length
        self.player_speed = 5 
        self.player_rot_speed = 5
        self.num_threads = num_threads
        self.biased_odds = 0.1
        self.augmented = True
        
        # New: Batch processing parameters
        self.batch_size = batch_size
        self.tensorized_game = TensorizedPseudoGame(
            batch_size=batch_size,
            device=device,
            min_num_sampled_waypoints=min_num_waypoints,
            max_num_sampled_waypoints=max_num_waypoints
        )
        
        # Pre-generated batch cache for efficiency
        self._batch_cache = []
        self._cache_lock = threading.Lock()
        
        # Thread-local storage for agent keypoints
        self._thread_local = threading.local()

    def get_batch_samples(self, batch_size: int) -> Tuple[List, List, torch.Tensor]:
        """
        Generate a batch of samples using efficient tensorized operations when possible,
        falling back to threaded generation for smaller batches.
        
        Returns:
            curr_obs_batch: List of batch_size current observations
            context_batch: List of batch_size contexts (each context is a list of demos)
            clean_actions_batch: Tensor of shape [batch_size, pred_horizon, 4]
        """
        
        # Use tensorized generation for larger batches, threading for smaller ones
        if batch_size >= self.batch_size and batch_size % self.batch_size == 0:
            return self._get_batch_samples_tensorized(batch_size)
        else:
            return self._get_batch_samples_threaded(batch_size)
    
    def _get_batch_samples_tensorized(self, batch_size: int) -> Tuple[List, List, torch.Tensor]:
        """Generate batch samples using efficient tensorized operations."""
        
        num_tensor_batches = batch_size // self.batch_size
        
        curr_obs_batch = []
        context_batch = []
        clean_actions_list = []
        
        for _ in range(num_tensor_batches):
            # Generate a full tensorized batch
            tensor_batch_data = self._generate_tensorized_batch()
            
            # Convert tensorized batch to the expected format
            batch_samples = self._convert_tensor_batch_to_samples(tensor_batch_data)
            
            for curr_obs, context, clean_actions in batch_samples:
                curr_obs_batch.append(curr_obs)
                context_batch.append(context)
                clean_actions_list.append(clean_actions)
        
        clean_actions_batch = torch.stack(clean_actions_list, dim=0)
        return curr_obs_batch, context_batch, clean_actions_batch
    
    def _get_batch_samples_threaded(self, batch_size: int) -> Tuple[List, List, torch.Tensor]:
        """Fallback to original threaded approach for non-aligned batch sizes."""
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._generate_single_sample) for _ in range(batch_size)]
            
            curr_obs_batch = []
            context_batch = []
            clean_actions_list = []
            
            for future in as_completed(futures):
                curr_obs, context, clean_actions = future.result()
                curr_obs_batch.append(curr_obs)
                context_batch.append(context)
                clean_actions_list.append(clean_actions)
        
        clean_actions_batch = torch.stack(clean_actions_list, dim=0)
        return curr_obs_batch, context_batch, clean_actions_batch
    
    def _generate_tensorized_batch(self) -> Dict:
        """Generate a batch of trajectories using tensorized operations."""
        
        # Determine how many should be biased
        biased_mask = torch.rand(self.batch_size) < self.biased_odds
        
        # Generate base batch data
        batch_data = self.tensorized_game.generate_batch_trajectories()
        
        # Apply biased/unbiased logic to different elements of the batch
        for i in range(self.batch_size):
            if biased_mask[i]:
                # Update this batch element to use biased generation
                self._apply_biased_generation(batch_data, i)
        
        return batch_data
    
    def _apply_biased_generation(self, batch_data: Dict, batch_idx: int):
        """Apply biased generation logic to a specific batch element."""
        # This would contain the logic from your original biased waypoint generation
        # Simplified for now - you can adapt your existing biased logic here
        pass
    
    def _convert_tensor_batch_to_samples(self, tensor_batch_data: Dict) -> List[Tuple]:
        """Convert tensorized batch data to individual samples in the expected format."""
        
        batch_samples = []
        waypoints = tensor_batch_data['waypoints']
        actions = tensor_batch_data['actions']
        observations = tensor_batch_data['observations']
        
        for b in range(self.batch_size):
            # Extract data for this batch element
            sample_waypoints = waypoints[b]
            sample_actions = actions[b]
            
            # Generate context demos for this sample
            context = self._generate_context_for_sample(sample_waypoints, b)
            
            # Process actions to match your expected format
            processed_actions = self._process_actions_tensor(sample_actions)
            
            # Create current observation (simplified - adapt as needed)
            curr_obs = self._create_current_observation(sample_waypoints, b)
            
            batch_samples.append((curr_obs, context, processed_actions))
        
        return batch_samples
    
    def _generate_context_for_sample(self, sample_waypoints: torch.Tensor, batch_idx: int) -> List:
        """Generate context demonstrations for a single sample."""
        context = []
        
        # For now, generate context using the original method
        # You could potentially tensorize this too for even more efficiency
        for _ in range(self.num_demos - 1):
            # Create a variation of the current trajectory
            context_demo = self._create_context_demo_from_waypoints(sample_waypoints)
            context.append(context_demo)
        
        return context
    
    def _create_context_demo_from_waypoints(self, waypoints: torch.Tensor) -> List:
        """Create a context demonstration from waypoints."""
        # Convert waypoints to observations format
        # This is a simplified version - adapt based on your observation structure
        
        observations = []
        for i in range(min(self.demo_length, waypoints.shape[0])):
            # Create proper agent position (center + 3 triangle points) = (4, 2)
            player_pos = waypoints[i, :2].unsqueeze(0)  # (1, 2)
            player_orientation = torch.zeros(1, device=self.device)  # (1,)
            
            # Get agent position using tensorized method
            agent_pos = self.tensorized_game._get_agent_pos_batch(player_pos, player_orientation)
            agent_pos = agent_pos.squeeze(0)  # Remove batch dimension: (4, 2)
            
            obs = {
                'point-clouds': torch.randn(100, 3),  # Placeholder
                'coords': torch.randn(100, 2),        # Placeholder  
                'agent-pos': agent_pos.cpu().numpy(),  # Now (4, 2) as expected
                'agent-state': waypoints[i, 2].cpu().numpy(),
                'agent-orientation': 0.0,  # Placeholder
                'done': i == waypoints.shape[0] - 1,
                'time': i
            }
            observations.append(obs)
        
        return observations
    
    def _process_actions_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        """Process actions to match the expected SE2 + state format."""
        num_actions = min(self.demo_length, actions.shape[0])
        processed_actions = torch.zeros(num_actions, 10, device=self.device)  # 9 for SE2 + 1 for state
        
        # Keep track of cumulative orientation for proper SE(2) conversion
        current_orientation = 0.0  # Starting orientation in degrees
        
        for i in range(num_actions):
            forward_movement = actions[i, 0]    # Distance magnitude
            rotation_deg = actions[i, 1]        # Rotation angle in degrees  
            state = actions[i, 2]               # State change
            
            # Update orientation: new orientation after this rotation
            current_orientation += rotation_deg
            
            # Convert to radians for trigonometry
            total_orientation_rad = current_orientation * torch.pi / 180.0
            rotation_rad = rotation_deg * torch.pi / 180.0
            
            # Option B: Forward movement is in the direction of FINAL orientation
            # Decompose forward movement into x,y components
            x_translation = forward_movement * torch.cos(total_orientation_rad)
            y_translation = forward_movement * torch.sin(total_orientation_rad)
            
            # Create SE(2) matrix for this action
            # This represents the transformation: rotate THEN translate
            cos_r = torch.cos(rotation_rad)
            sin_r = torch.sin(rotation_rad)
            
            se2_matrix = torch.zeros(3, 3, device=self.device)
            se2_matrix[0, 0] = cos_r      # Rotation component
            se2_matrix[0, 1] = -sin_r     # Rotation component
            se2_matrix[1, 0] = sin_r      # Rotation component  
            se2_matrix[1, 1] = cos_r      # Rotation component
            se2_matrix[0, 2] = x_translation  # x translation in world frame
            se2_matrix[1, 2] = y_translation  # y translation in world frame
            se2_matrix[2, 2] = 1.0        # Homogeneous coordinate
            
            processed_actions[i, :9] = se2_matrix.flatten()
            processed_actions[i, 9] = state
        
        # Apply accumulation to get cumulative transformations
        processed_actions = self._accumulate_actions_tensor(processed_actions)
        
        return processed_actions

    
    def _accumulate_actions_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        """Tensorized version of action accumulation."""
        n = actions.shape[0]
        
        # Extract and reshape SE(2) matrices
        se2_matrices = actions[:, :9].view(n, 3, 3)
        state_actions = actions[:, 9:]
        
        # Compute cumulative matrix products using efficient tensor operations
        cumulative_matrices = torch.zeros_like(se2_matrices)
        cumulative_matrices[0] = se2_matrices[0]
        
        for i in range(1, n):
            cumulative_matrices[i] = torch.matmul(cumulative_matrices[i-1], se2_matrices[i])
        
        # Flatten back and concatenate with state actions
        cumulative_se2_flat = cumulative_matrices.view(n, 9)
        cumulative_actions = torch.cat([cumulative_se2_flat, state_actions], dim=1)
        
        return cumulative_actions
    
    def _create_current_observation(self, waypoints: torch.Tensor, batch_idx: int) -> Dict:
        """Create current observation for a sample with proper agent position."""
        # Create a single player position and orientation for this sample
        player_pos = waypoints[0, :2].unsqueeze(0)  # (1, 2)
        player_orientation = torch.zeros(1, device=self.device)  # (1,) - default orientation
        
        # Calculate agent position (center + triangle keypoints) using the tensorized method
        # This requires access to the tensorized game's method
        agent_pos = self.tensorized_game._get_agent_pos_batch(player_pos, player_orientation)
        agent_pos = agent_pos.squeeze(0)  # Remove batch dimension: (4, 2)
        
        return {
            'point-clouds': torch.randn(100, 3),  # Placeholder
            'coords': torch.randn(100, 2),        # Placeholder
            'agent-pos': agent_pos.cpu().numpy(),  # Now (4, 2) as expected
            'agent-state': waypoints[0, 2].cpu().numpy(), 
            'agent-orientation': 0.0,  # Placeholder
            'done': False,
            'time': 0
        }
    
    def _generate_single_sample(self) -> Tuple[dict, List, torch.Tensor]:
        """Original single sample generation (kept for compatibility)."""
        biased = np.random.rand() < self.biased_odds
        augmented = self.augmented
        pseudo_game = self._make_game(biased, augmented)
        context = self._get_context(pseudo_game)   
        curr_obs, clean_actions = self._get_ground_truth(pseudo_game)
        return curr_obs, context, clean_actions
    
    def get_agent_keypoints(self):
        """Original method preserved."""
        agent_keypoints = torch.zeros((len(self.agent_key_points), 2), device=self.device)
        agent_keypoints[0] = torch.tensor(self.agent_key_points['front'], device=self.device)
        agent_keypoints[1] = torch.tensor(self.agent_key_points['back-left'], device=self.device)
        agent_keypoints[2] = torch.tensor(self.agent_key_points['back-right'], device=self.device)
        agent_keypoints[3] = torch.tensor(self.agent_key_points['center'], device=self.device)
        return agent_keypoints
    
    def _make_game(self, biased, augmented):
        """Original method preserved."""
        # Get screen dimensions from configs or use defaults
        screen_width = PSEUDO_SCREEN_WIDTH
        screen_height = PSEUDO_SCREEN_HEIGHT
        player_starting_pos = (random.randint(0, screen_width), random.randint(0, screen_height))

        return PseudoGame(
            player_starting_pos=player_starting_pos,
            max_num_sampled_waypoints=self.max_num_waypoints, 
            min_num_sampled_waypoints=self.min_num_waypoints, 
            biased=biased, 
            augmented=augmented
        )
    
    def _run_game(self, pseudo_demo):
        """Original method preserved."""
        max_retries = 1000
        for attempt in range(max_retries):
            try: 
                pseudo_demo.reset_game(shuffle=True)
                pseudo_demo.run()
                return pseudo_demo
            except Exception as e:
                if attempt == max_retries-1:
                    raise 
                continue
    
    def _get_context(self, pseudo_game):
        """Original method preserved."""
        context = []
        for _ in range(self.num_demos - 1):
            pseudo_demo = self._run_game(pseudo_game)
            observations = pseudo_demo.observations
            sample_rate = len(observations) // self.demo_length
            sampled_obs = observations[::sample_rate][:self.demo_length]
            context.append(sampled_obs)
        return context
    
    def _get_ground_truth(self, pseudo_game):
        """Original method preserved."""
        pseudo_game.set_augmented(np.random.rand() > 0.5) 
        pseudo_demo = self._run_game(pseudo_game)
        pd_actions = pseudo_demo.get_actions(mode='se2')

        se2_actions = np.array([action[0].flatten() for action in pd_actions])
        state_actions = np.array([action[1] for action in pd_actions])
        state_actions = state_actions.reshape(-1,1)
        actions = np.concatenate([se2_actions, state_actions], axis=1)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)          
        
        actions = self._accumulate_actions(actions)
        sample_rate = actions.shape[0] // self.demo_length
        actions = actions[::sample_rate][:self.demo_length]
        true_obs = pseudo_demo.observations[::sample_rate][:self.demo_length]

        return true_obs[0], actions
    
    def _accumulate_actions(self, actions):
        """Original method preserved."""
        n = actions.shape[0]
        
        se2_matrices = actions[:, :9].view(n, 3, 3)
        state_actions = actions[:, 9:]
        
        cumulative_matrices = torch.zeros_like(se2_matrices)
        cumulative_matrices[0] = se2_matrices[0]
        
        for i in range(1, n):
            cumulative_matrices[i] = torch.matmul(cumulative_matrices[i-1], se2_matrices[i])
        
        cumulative_se2_flat = cumulative_matrices.view(n, 9)
        cumulative_actions = torch.cat([cumulative_se2_flat, state_actions], dim=1)
        
        return cumulative_actions

# Usage example - drop-in replacement for your original generator
def create_efficient_demo_generator(device='cpu', batch_size=32, **kwargs):
    """Create a tensorized demo generator that's a drop-in replacement."""
    return TensorizedPseudoDemoGenerator(
        device=device,
        batch_size=batch_size,
        **kwargs
    )


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
        # clean_actions_batch = torch.stack(clean_actions_list, dim=0)
        clean_actions_batch = clean_actions_list
        
        return curr_obs_batch, context_batch, clean_actions_batch

    def _generate_single_sample(self) -> Tuple[dict, List, torch.Tensor]:
        """Generate a single training sample (thread-safe)"""
        biased = np.random.rand() < self.biased_odds
        augmented = self.augmented # for now 
        pseudo_game = self._make_game(biased, augmented)
        context = self._get_context(pseudo_game)   
        curr_obs, clean_actions = self._get_ground_truth(pseudo_game)
        return curr_obs, context, clean_actions

    def get_agent_keypoints(self):
    
        agent_keypoints = torch.zeros((len(self.agent_key_points), 2), device=self.device)
        agent_keypoints[0] = torch.tensor(self.agent_key_points['front'], device=self.device)
        agent_keypoints[1] = torch.tensor(self.agent_key_points['back-left'], device=self.device)
        agent_keypoints[2] = torch.tensor(self.agent_key_points['back-right'], device=self.device)
        agent_keypoints[3] = torch.tensor(self.agent_key_points['center'], device=self.device)
        return agent_keypoints
    
    def _make_game(self, biased,augmented):
        player_starting_pos =(random.randint(0,PSEUDO_SCREEN_WIDTH), random.randint(0,PSEUDO_SCREEN_HEIGHT))
        return PseudoGame(
                    player_starting_pos=player_starting_pos,
                    max_num_sampled_waypoints=self.max_num_waypoints, 
                    min_num_sampled_waypoints=self.min_num_waypoints, 
                    biased=biased, 
                    augmented=augmented
                )

    def _run_game(self, pseudo_demo):
        max_retries = 1000
        player_starting_pos =(random.randint(0,PSEUDO_SCREEN_WIDTH), random.randint(0,PSEUDO_SCREEN_HEIGHT))
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
            sampled_obs = self._downsample_obs(observations)
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
        sample_rate = min(actions.shape[0] // self.demo_length,1)
        assert temp == actions.shape
        actions = self._downsample_actions(actions)
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

    def _downsample_actions(self,actions):
        if actions.shape[0] < self.demo_length:
            return actions
        result = torch.zeros((10, actions.shape[1]), device = actions.device)
        result[0] = actions[0]
        total_middle_positions = actions.shape[0] - 2  # Exclude first and last
        
        for i in range(1, self.demo_length - 1):
            # Calculate the position in the middle section (0 to total_middle_positions-1)
            middle_pos = (i - 1) * total_middle_positions / (self.demo_length - 3) if self.demo_length > 3 else total_middle_positions // 2
            # Convert to actual index (add 1 to skip first item)
            actual_index = int(round(middle_pos)) + 1
            result[i] = actions[actual_index]
        
        result[-1] = actions[-1]  # Always include last item
        
        return result
    
    def _downsample_obs(self, observations):
        if len(observations) < self.demo_length:
            return observations

        result = [observations[0]]
        total_middle_positions = len(observations) - 2  # Exclude first and last
        
        for i in range(1, self.demo_length - 1):
            # Calculate the position in the middle section (0 to total_middle_positions-1)
            middle_pos = (i - 1) * total_middle_positions / (self.demo_length - 3) if self.demo_length > 3 else total_middle_positions // 2
            # Convert to actual index (add 1 to skip first item)
            actual_index = int(round(middle_pos)) + 1
            result.append(observations[actual_index])
        
        result.append(observations[-1])  # Always include last item
        
        return result

# Example usage:
# Old way:
# demo_gen = PseudoDemoGenerator(device='cuda', num_demos=5)
# curr_obs, context, actions = demo_gen.get_batch_samples(64)

# New way (same API, but faster):
# demo_gen = TensorizedPseudoDemoGenerator(device='cpu', num_demos=5, batch_size=32)
# curr_obs, context, actions = demo_gen.get_batch_samples(64)  # Will use tensorized path