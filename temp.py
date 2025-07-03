import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

class InstantPolicyTrainer:
    def __init__(self, agent, device='cuda'):
        self.agent = agent
        self.device = device
        
        # Diffusion parameters (from the paper)
        self.num_diffusion_steps = 1000  # K in equations
        self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)
        
        # Optimizer (from appendix: AdamW with 1e-5 learning rate)
        self.optimizer = optim.AdamW(agent.parameters(), lr=1e-5)
        
        # Action normalization bounds (from appendix)
        self.max_translation = 0.01  # 1cm
        self.max_rotation = np.pi / 60  # 3 degrees
        
    def _linear_beta_schedule(self, beta_start, beta_end, timesteps):
        """Create linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def train_step(self, pseudo_demonstrations: Dict) -> float:
        """
        Single training step implementing the diffusion loss
        
        Args:
            pseudo_demonstrations: Dict containing:
                - context: List of demonstration trajectories (N-1 demos)
                - target_trajectory: The trajectory to predict actions for
                - current_observations: Point cloud observations
                - ground_truth_actions: Target actions to learn
        """
        
        # Step 1: Sample random timestep k for diffusion
        batch_size = len(pseudo_demonstrations['ground_truth_actions'])
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        
        # Step 2: Add noise to ground truth actions (Equation 1)
        noisy_actions, noise = self._add_noise_to_actions(
            pseudo_demonstrations['ground_truth_actions'], 
            timesteps
        )
        
        # Step 3: Construct graph representation with noisy actions
        graph_data = self._construct_graph(
            context=pseudo_demonstrations['context'],
            current_obs=pseudo_demonstrations['current_observations'],
            actions=noisy_actions  # Use noisy actions in graph
        )
        
        # Step 4: Predict noise/denoising directions
        predicted_noise = self.agent(graph_data, timesteps)
        
        # Step 5: Compute loss (MSE between predicted and actual noise)
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # Step 6: Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _add_noise_to_actions(self, actions: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to actions according to diffusion forward process (Equation 1)
        
        Args:
            actions: Ground truth actions [batch_size, T, 7] (SE(3) + gripper)
            timesteps: Diffusion timesteps [batch_size]
        
        Returns:
            noisy_actions: Actions with added noise
            noise: The noise that was added (target for prediction)
        """
        
        # Separate SE(3) transformations and gripper actions
        se3_actions = actions[..., :6]  # Translation (3) + rotation (3)
        gripper_actions = actions[..., 6:7]  # Binary gripper state
        
        # Project SE(3) to se(3) tangent space for noise addition
        se3_tangent = self._se3_to_tangent(se3_actions)
        
        # Normalize to [-1, 1] range (from appendix)
        se3_normalized = self._normalize_se3(se3_tangent)
        
        # Sample noise
        se3_noise = torch.randn_like(se3_normalized)
        gripper_noise = torch.randn_like(gripper_actions)
        
        # Add noise according to schedule: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1, 1)
        
        noisy_se3 = torch.sqrt(alpha_cumprod_t) * se3_normalized + torch.sqrt(1 - alpha_cumprod_t) * se3_noise
        noisy_gripper = torch.sqrt(alpha_cumprod_t[..., 0]) * gripper_actions + torch.sqrt(1 - alpha_cumprod_t[..., 0]) * gripper_noise
        
        # Convert back to SE(3) for graph construction
        noisy_se3_unnorm = self._unnormalize_se3(noisy_se3)
        noisy_se3_actions = self._tangent_to_se3(noisy_se3_unnorm)
        
        noisy_actions = torch.cat([noisy_se3_actions, noisy_gripper], dim=-1)
        noise = torch.cat([se3_noise, gripper_noise], dim=-1)
        
        return noisy_actions, noise
    
    def _construct_graph(self, context: list, current_obs: Dict, actions: torch.Tensor) -> Dict:
        """
        Construct the heterogeneous graph representation G(G_l^a(a), G_c(G_l^t, {G_l^{1:L}}_1^N))
        """
        
        # Process point cloud observations into node features
        scene_features = self._process_point_clouds(current_obs['point_clouds'])
        
        # Create gripper nodes for current state
        current_gripper_features = self._create_gripper_nodes(
            current_obs['end_effector_pose'],
            current_obs['gripper_state']
        )
        
        # Create gripper nodes for demonstrations
        demo_gripper_features = []
        for demo in context:
            demo_features = self._create_gripper_nodes(
                demo['end_effector_poses'],
                demo['gripper_states']
            )
            demo_gripper_features.append(demo_features)
        
        # Create action nodes from (potentially noisy) actions
        action_gripper_features = self._create_action_nodes(actions)
        
        # Combine all node features
        node_features = {
            'scene_point': scene_features,
            'gripper_current': current_gripper_features,
            'gripper_demo': torch.cat(demo_gripper_features, dim=0),
            'gripper_action': action_gripper_features
        }
        
        # Create edges between different node types
        edge_indices, edge_features = self._create_edges(
            scene_features, current_gripper_features, 
            demo_gripper_features, action_gripper_features
        )
        
        # Create node type indices
        node_types_per_node = self._create_node_type_indices(node_features)
        
        return {
            'node_features': node_features,
            'edge_indices': edge_indices,
            'edge_features': edge_features,
            'node_types_per_node': node_types_per_node
        }
    
    def train_epoch(self, pseudo_demo_generator, num_steps_per_epoch=10000):
        """
        Train for one epoch using continuously generated pseudo-demonstrations
        """
        total_loss = 0.0
        
        for step in range(num_steps_per_epoch):
            # Generate new pseudo-demonstrations on the fly (Section 3.4)
            pseudo_demos = pseudo_demo_generator.generate_batch()
            
            # Train step
            loss = self.train_step(pseudo_demos)
            total_loss += loss
            
            if step % 1000 == 0:
                print(f"Step {step}, Loss: {loss:.6f}")
        
        return total_loss / num_steps_per_epoch
    
    def full_training(self, pseudo_demo_generator, num_epochs=2500000//10000):
        """
        Full training loop (2.5M optimization steps as mentioned in paper)
        """
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(pseudo_demo_generator)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
            
            # Learning rate cooldown in final 50k steps
            if epoch > num_epochs - 50:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.99


class PseudoDemonstrationGenerator:
    """
    Generates pseudo-demonstrations as described in Section 3.4
    """
    def __init__(self, shapenet_objects, num_demos_per_task=2, demo_length=10):
        self.shapenet_objects = shapenet_objects
        self.num_demos_per_task = num_demos_per_task
        self.demo_length = demo_length
    
    def generate_batch(self, batch_size=32):
        """
        Generate a batch of pseudo-demonstrations
        
        Returns:
            Dict containing context demos and target trajectory
        """
        batch = {
            'context': [],
            'current_observations': [],
            'ground_truth_actions': []
        }
        
        for _ in range(batch_size):
            # Sample random objects from ShapeNet
            objects = self._sample_objects()
            
            # Generate pseudo-task (sequence of waypoints)
            waypoints = self._generate_waypoints(objects)
            
            # Generate N demonstrations for this pseudo-task
            demonstrations = []
            for _ in range(self.num_demos_per_task + 1):  # +1 for target
                demo = self._generate_demonstration(objects, waypoints)
                demonstrations.append(demo)
            
            # Use N-1 demonstrations as context, last one as target
            batch['context'].append(demonstrations[:-1])
            batch['current_observations'].append(demonstrations[-1]['observations'])
            batch['ground_truth_actions'].append(demonstrations[-1]['actions'])
        
        return batch
    
    def _generate_demonstration(self, objects, waypoints):
        """
        Generate a single pseudo-demonstration by moving gripper between waypoints
        """
        # Randomly place objects in scene
        object_poses = self._randomize_object_poses(objects)
        
        # Choose random starting gripper pose
        start_pose = self._sample_gripper_start_pose()
        
        # Interpolate between waypoints to create trajectory
        trajectory = self._interpolate_trajectory(start_pose, waypoints)
        
        # Render point clouds and record gripper states
        observations = []
        actions = []
        
        for i, pose in enumerate(trajectory):
            # Render scene from gripper perspective
            point_cloud = self._render_point_cloud(object_poses, pose)
            
            obs = {
                'point_clouds': point_cloud,
                'end_effector_pose': pose[:6],  # SE(3)
                'gripper_state': pose[6]  # Binary
            }
            observations.append(obs)
            
            # Compute action to next waypoint
            if i < len(trajectory) - 1:
                action = self._compute_action(pose, trajectory[i+1])
                actions.append(action)
        
        return {
            'observations': observations,
            'actions': torch.tensor(actions, dtype=torch.float32)
        }


# Usage example:
if __name__ == "__main__":
    # Initialize agent
    agent = InstantPolicyAgent()
    
    # Initialize trainer
    trainer = InstantPolicyTrainer(agent)
    
    # Initialize pseudo-demonstration generator
    pseudo_gen = PseudoDemonstrationGenerator(shapenet_objects=[])
    
    # Train the model
    trainer.full_training(pseudo_gen)