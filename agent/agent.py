"""
ALL 3 NETWORKS ARE HETERO GRAPH TRANSFORMERS with 2 layers and a hidden dimension of size 1024 
(16 heads, each with 64 dimensions)
    F′_i = W_1 * F_i+ att_i,j(W_2F_j + W_5e_ij) ; 
    att_i,j= softmax((W_3F_i)^T (W_4F_j + W_5e_ij)/√d)
    d is a scaling factor for diffusion stability 


Since it is hetero graphs, each node and edge type are processed with separate learnable weights and 
agg via summation to produce node-wise embeddings 

This can be understood as a set of cross-attention mechanisms

Layer Normalisation layers is used between every attention layer 
Additional residual connections is used to ensure good propagation of gradients throughout network 

Finally features of the nodes representing robot actions are processed with a 2-layer MLP equipped with
GeLU to produce per-node denoising directions

   

We assume node feature to be represented by their coordinates (x,y) and their node type. 
For edible node, there is an extra feature 'eaten'

We also need to think about positional encoding for the coordinates of

so there is 2 types of nodes : (3 x 1) and (4 x 1)



Modules each have 5 kinds of weights: 
    W1 : num_nodes x num_node_feature
    W2 : num_nodes x num_node_feature
    W3 : num_nodes x num_node_feature
    W4 : num_nodes x num_node_feature
    W5 : num_edges x num_edge_feature

    each node type have their own W1-W4 
    each edge typehave their own W5 

    4 node types in total, edge types will be discussed individually
"""
import torch
import torch.nn as nn

import numpy as np
from enum import Enum

# from ..utils.graph import DemoGraph, ContextGraph, ActionGraph, EdgeType, NodeType, make_localgraph # use when running this file
from utils.graph import LocalGraph, DemoGraph, ContextGraph, ActionGraph, EdgeType, NodeType # use when running from rwd

# from ..tasks.twoD.game import Action # use when running this file
from tasks.twoD.game import Action, BLACK, YELLOW, GREEN, RED# use when running from rwd

from typing import Dict, List, Tuple

from agent.geometry_encoder import GeometryEncoder2D 
from collections import defaultdict
"""
# ok unless we make actions stack this dont make sense hmm
# need to produce per node denoising directions
# right now from psd-gen we get curr_obs, clean_actions 
# we then add noise to action by:
    # first from (x,y,theta, sc) => SE(2) (3x3 mat) + sc
    # SE(2) => se(2) tangent space (3x1 vector)
    # 
# so our label now becomes noise (caused by noisy actions) on future node positions 
# so right now the code runs as follows:
    # with curr obs, noisy action and context
    # context agg 
    # noisy actions stacks and applied to current graph 
    # node emb for future nodes produced
    # self.pred_p to predict the noise added in future positions
    # self.pred_g to predict the noise added in state

# to get noise added in future state from noisy actions:
    #prev code to add noise  
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
    #     noisy_se2 = torch.sqrt(alpha_cumprod_t) * se2_normalized + torch.sqrt(1 - alpha_cumprod_t) * se2_noise
    #     noisy_binary = torch.sqrt(alpha_cumprod_t) * binary_actions + torch.sqrt((1 - alpha_cumprod_t[..., 0]).unsqueeze(-1)) * binary_noise
        
    #     # Convert back to SE(2) for graph construction
    #     noisy_se2_unnorm = self._unnormalize_se2(noisy_se2)
    #     noisy_se2_actions = self._tangent_to_se2(noisy_se2_unnorm)
        
    #     noisy_actions = torch.cat([noisy_se2_actions, noisy_binary], dim=-1)
    #     noise = torch.cat([se2_noise, binary_noise], dim=-1)
    #     return noisy_actions, noise.float()
    
    # def _se2_to_tangent(self, se2_actions):
    #     #Convert SE(2) [x, y, theta] to se(2) tangent space
    #     # For SE(2), tangent space is just [x, y, theta] since it's already linear
    #     return se2_actions

    # def _tangent_to_se2(self, tangent):
    #     #Convert se(2) tangent space back to SE(2)
    #     return tangent

    # def _normalize_se2(self, se2_tangent):
    #     # Normalize SE(2) tangent vectors to [-1, 1]
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
    #     # Unnormalize SE(2) from [-1, 1] back to original ranges
    #     translation = normalized_se2[..., :2] * self.translation_scale
    #     rotation = normalized_se2[..., 2:3] * torch.pi
        
    #     return torch.cat([translation, rotation], dim=-1)

"""

class InstantPolicyAgent(nn.Module):

    def __init__(self,
                device,
                max_translation,
                num_diffusion_steps = 100,
                num_agent_nodes = 4, 
                pred_horizon = 5, 
                hidden_dim = 64, 
                node_embd_dim = 16,
                edge_embd_dim = 16,
                agent_state_embd_dim = 4,
                edge_pos_dim = 2
                ):
        super(InstantPolicyAgent, self).__init__()
        
        self.policy = InstantPolicy(
                device=device, 
                num_agent_nodes=num_agent_nodes,
                pred_horizon=pred_horizon,
                hidden_dim=hidden_dim,
                node_embd_dim=node_embd_dim,
                edge_embd_dim=edge_embd_dim,
                agent_state_embd_dim=agent_state_embd_dim,
                edge_pos_dim=edge_pos_dim
                )
        self.max_translation = max_translation
        self.device = device 
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = self._linear_beta_schedule(0.0001, 0.02, self.num_diffusion_steps)
        self.alpha_schedule = 1-self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)
        self.hidden_dim = hidden_dim
        self.node_embd_dim = node_embd_dim

        self.pred_head_p = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,2, device=self.device),
        )

        self.pred_head_g = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim,device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,1,device=self.device),
            nn.ReLU(),
        )
        

    # right now action is x,y,theta,state_change
    # 
    # optimally want to produce:  
    # x_cetner_mass_trans, y_centre_mass_trans, 
    def forward(self,
                curr_obs,
                context,
                clean_actions):
        
        # need to use clean actions to generate noisy actions 
        batch_size = len(clean_actions)
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        noisy_actions, noise = self._get_noisy_actions(clean_actions, timesteps)
        # actual_noise = self._get_noise(noise)

        node_embs = self.policy(curr_obs, context, noisy_actions) # N x self.num_agent_nodes x self.node_emb_dim
        _p_deltas = self.pred_head_p(node_embs) # N x self.num_agent_nodes x 2 
        _g_deltas = self.pred_head_g(node_embs) # N x self.num_agent_nodes x 1

        # actual noise needs to be N x self.num_agent_nodes x 3
        # _p_deltas is (N x num-agent-nodes x 2, _g_deltas )
        predicted_noise = torch.cat([_p_deltas, _g_deltas], dim=-1)

        return predicted_noise, noise 

    def _linear_beta_schedule(self, beta_start, beta_end, timesteps):
        """Create linear noise schedule"""
        return torch.linspace(beta_start, beta_end, timesteps, device = self.device)
    
    def _SE2_to_actions(self,se2):
        # actions from 3x3 matrix transformation => (x,y,theta)
        num_actions = se2.shape[0]
        actions = torch.zeros(num_actions, 3, device=self.device)
        
        # Extract translation components
        actions[:, 0] = se2[:, 0, 2]  # x_trans
        actions[:, 1] = se2[:, 1, 2]  # y_trans
        
        # Extract rotation angle and convert to degrees
        angles_rad = torch.atan2(se2[:, 1, 0], se2[:, 0, 0])
        actions[:, 2] = torch.rad2deg(angles_rad)
        
        return actions

    def _actions_to_SE2(self,actions):
        num_actions = actions.shape[0]
        se2_actions = torch.zeros(num_actions, 3,3, device=self.device) # 2x2 rot matr, with 2x1 translation mat
        angles = torch.deg2rad(actions[:, 2])
        x_trans = actions[:, 0]
        y_trans = actions[:, 1]

        se2_actions[:, 0, 2] = x_trans
        se2_actions[:, 1, 2] = y_trans
        se2_actions[:, 2, 2] = 1

        _sin_theta = torch.sin(angles)
        _cos_theta = torch.cos(angles)
        se2_actions[:, 0, 0] = _cos_theta
        se2_actions[:, 0, 1] = - _sin_theta
        se2_actions[:, 1, 0] = _sin_theta
        se2_actions[:, 1, 1] = _cos_theta

        return se2_actions

    def _SE2_to_se2(self,T, eps=1e-6):
        # Handle batch dimensions
        batch_shape = T.shape[:-2]
        
        # Extract rotation matrix (top-left 2x2)
        R = T[..., :2, :2]
        
        # Extract translation vector (top-right 2x1)
        t = T[..., :2, 2]
        
        # Compute rotation angle using atan2
        theta = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        
        # Compute V_inverse matrix for translation
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # Handle small angle case for numerical stability
        small_angle_mask = torch.abs(theta) < eps
        
        # V_inverse for general case
        V_inv = torch.zeros(*batch_shape, 2, 2, device=T.device, dtype=T.dtype)
        
        # For non-small angles
        not_small = ~small_angle_mask
        if torch.any(not_small):
            theta_nz = theta[not_small]
            sin_nz = sin_theta[not_small]
            cos_nz = cos_theta[not_small]
            
            # V_inv = (1/theta) * [[sin(theta), cos(theta)-1], [1-cos(theta), sin(theta)]]
            V_inv[not_small, 0, 0] = sin_nz / theta_nz
            V_inv[not_small, 0, 1] = (cos_nz - 1) / theta_nz
            V_inv[not_small, 1, 0] = (1 - cos_nz) / theta_nz
            V_inv[not_small, 1, 1] = sin_nz / theta_nz
        
        # For small angles, V_inv ≈ I - (1/2) * [[0, -theta], [theta, 0]]
        if torch.any(small_angle_mask):
            V_inv[small_angle_mask, 0, 0] = 1.0
            V_inv[small_angle_mask, 1, 1] = 1.0
            V_inv[small_angle_mask, 0, 1] = -0.5 * theta[small_angle_mask]
            V_inv[small_angle_mask, 1, 0] = 0.5 * theta[small_angle_mask]
        
        # Compute rho = V_inv @ t
        rho = torch.einsum('...ij,...j->...i', V_inv, t)
        
        # Stack to form se(2) vector [rho_x, rho_y, theta]
        xi = torch.stack([rho[..., 0], rho[..., 1], theta], dim=-1)
        
        return xi

    def _se2_to_SE2(self,xi):
        """
        Convert se(2) tangent vector to SE(2) matrix using exponential map.
        
        Args:
            xi: se(2) tangent vector of shape (..., 3) as [rho_x, rho_y, theta]
        
        Returns:
            T: SE(2) transformation matrix of shape (..., 3, 3)
        """
        batch_shape = xi.shape[:-1]
        
        rho = xi[..., :2]  # [rho_x, rho_y]
        theta = xi[..., 2]  # theta
        
        # Compute rotation matrix
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        R = torch.zeros(*batch_shape, 2, 2, device=xi.device, dtype=xi.dtype)
        R[..., 0, 0] = cos_theta
        R[..., 0, 1] = -sin_theta
        R[..., 1, 0] = sin_theta
        R[..., 1, 1] = cos_theta
        
        # Compute V matrix for translation
        eps = 1e-6
        small_angle_mask = torch.abs(theta) < eps
        
        V = torch.zeros(*batch_shape, 2, 2, device=xi.device, dtype=xi.dtype)
        
        # For non-small angles
        not_small = ~small_angle_mask
        if torch.any(not_small):
            theta_nz = theta[not_small]
            sin_nz = sin_theta[not_small]
            cos_nz = cos_theta[not_small]
            
            # V = (1/theta) * [[sin(theta), -(1-cos(theta))], [1-cos(theta), sin(theta)]]
            V[not_small, 0, 0] = sin_nz / theta_nz
            V[not_small, 0, 1] = -(1 - cos_nz) / theta_nz
            V[not_small, 1, 0] = (1 - cos_nz) / theta_nz
            V[not_small, 1, 1] = sin_nz / theta_nz
        
        # For small angles, V ≈ I + (1/2) * [[0, -theta], [theta, 0]]
        if torch.any(small_angle_mask):
            V[small_angle_mask, 0, 0] = 1.0
            V[small_angle_mask, 1, 1] = 1.0
            V[small_angle_mask, 0, 1] = -0.5 * theta[small_angle_mask]
            V[small_angle_mask, 1, 0] = 0.5 * theta[small_angle_mask]
        
        # Compute translation: t = V @ rho
        t = torch.einsum('...ij,...j->...i', V, rho)
        
        # Construct SE(2) matrix
        T = torch.zeros(*batch_shape, 3, 3, device=xi.device, dtype=xi.dtype)
        T[..., :2, :2] = R
        T[..., :2, 2] = t
        T[..., 2, 2] = 1.0
        
        return T

    # for large displacement, project to SE2 straight up, no need for se2
    # our task involves 
    def _get_noisy_actions(self, clean_actions, timesteps, mode = 'large'):

        # binary actions dealt with similarly
        binary_actions = clean_actions[..., 3:4]  # Binary agent state
        moving_actions = clean_actions[..., :-1]
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1)


        # need to read the 2023 Ed paper for this
        # To calculate a set of transformations Tnoise we use a combination of Langevin Dynamics (described
        # in Section C.4) and uniform sampling at different scales. We start the training with uniform sam-
        # pling in ranges of [−0.8,0.8] metres for translation and [−π,π] radians for rotation. After Nnumber
        # of optimisation steps (10K in our implementation), we incorporate Langevin Dynamics sampling
        # which we perform every 5 optimisation steps. During this phase, we also reduce the uniform sam-
        # pling range to [−0.1,0.1] metres for translation and [−π/4,π/4] radians for rotation. Although
        # creating negative samples using only Langevin Dynamics is sufficient, in practice, we found that
        # our described sampling strategy leads to faster convergence and more stable training for this specific
        # application of energy-based models.
        if mode == 'large':
            # project to SE2
            SE2_action = self._actions_to_SE2(moving_actions)

            # sample noise
            SE2_noise = torch.randn_like(SE2_action)
            binary_noise = torch.rand_like(binary_actions)

            # reshape alpha_cumprod for SE2 
            alpha_cumprod_t_reshaped = alpha_cumprod_t.view(-1, 1, 1)  # Make it (batch, 1, 1)
            
            # add noise acc to schedule
            noisy_se2 = torch.sqrt(alpha_cumprod_t_reshaped) * SE2_action + torch.sqrt(1 - alpha_cumprod_t_reshaped) * SE2_noise
            noisy_binary = torch.sqrt(alpha_cumprod_t) * binary_actions + torch.sqrt((1 - alpha_cumprod_t[..., 0]).unsqueeze(-1)) * binary_noise

            # convert back to actions for graph construction
            noisy_actions = self._SE2_to_actions(noisy_se2)
            movement_noise = self._SE2_to_actions(SE2_noise)

            full_noisy_actions = torch.cat([noisy_actions, noisy_binary], dim=-1)
            full_noise = torch.cat([movement_noise, binary_noise], dim=-1)
            return full_noisy_actions, full_noise.float()

        # small displacement treatment
        else:
            # first project action to SE(2), then to se(2)
            SE2_actions = self._actions_to_SE2(moving_actions)
            # SE2_actions = self._actions_to_SE2(clean_actions)
            se2_actions = self._SE2_to_se2(SE2_actions)
            # then normalise resulting vectors
            se2_normalised = self._normalise_se2(se2_actions)

            # sample noise
            se2_noise = torch.randn_like(se2_normalised)
            binary_noise = torch.randn_like(binary_actions)
            
            # add noise according to schedule
            noisy_se2 = torch.sqrt(alpha_cumprod_t) * se2_normalised + torch.sqrt(1 - alpha_cumprod_t) * se2_noise
            noisy_binary = torch.sqrt(alpha_cumprod_t) * binary_actions + torch.sqrt((1 - alpha_cumprod_t[..., 0]).unsqueeze(-1)) * binary_noise

            # Convert back noisy actions and noise to actions for graph construction
            noisy_se2_unnorm = self._unnormalize_se2(noisy_se2)
            noisy_se2_actions = self._se2_to_SE2(noisy_se2_unnorm)
            noisy_actions = self._SE2_to_actions(noisy_se2_actions)

            se2_noise_unorm = self._unnormalize_se2(noisy_se2)
            SE2_noise = self._se2_to_SE2(se2_noise_unorm)
            movement_noise = self._SE2_to_actions(SE2_noise)

            full_noisy_actions = torch.cat([noisy_actions, noisy_binary], dim=-1)
            full_noise = torch.cat([movement_noise, binary_noise], dim=-1)
            return full_noisy_actions, full_noise.float()

    def _normalise_se2(self,se2_actions):
        se2_actions[...,:2] /= self.max_translation
        se2_actions[...,2:3] /= torch.pi 
        return se2_actions

    def _unnormalize_se2(self, normalized_se2):
        """Unnormalize SE(2) from [-1, 1] back to original ranges"""
        translation = normalized_se2[..., :2] * self.max_translation
        rotation = normalized_se2[..., 2:3] * torch.pi
        
        return torch.cat([translation, rotation], dim=-1)



# TODO: add Pointnet++ and SA layer! for scene node embedding
def SinCosEdgeEmbedding(source, dest, device, D=3):
    num_feature = source.shape[0]
    embedding = torch.zeros((num_feature, 2 * D), device=device)
    diff = torch.tensor(dest - source)
    for d in range(D):
        sin_vals = torch.sin(2**d * torch.pi * diff)
        cos_vals = torch.cos(2**d * torch.pi * diff)
        embedding[:, 2*d] = sin_vals
        embedding[:, 2*d+1] = cos_vals
    
    return embedding



# add SA layer here
class InstantPolicy(nn.Module):
    
    def __init__(self, 
                device,
                num_agent_nodes = 4, 
                pred_horizon = 5, 
                hidden_dim = 64, 
                node_embd_dim = 16,
                edge_embd_dim = 16,
                agent_state_embd_dim = 4,
                edge_pos_dim = 2):
        super(InstantPolicy, self).__init__()

        self.device = device
        self.num_agent_nodes = num_agent_nodes
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.node_embd_dim = node_embd_dim
        self.edge_embd_dim = edge_embd_dim
        self.agent_state_embd_dim = agent_state_embd_dim

        # embedders
        self.agent_embedder = nn.Embedding(
            self.num_agent_nodes * self.pred_horizon,
            self.node_embd_dim - self.agent_state_embd_dim, 
            device=self.device
        )
        self.agent_state_embedder = nn.Linear(1, self.agent_state_embd_dim, device=self.device)
        self.spatial_edge_embedding = lambda source_pos, dest_pos : SinCosEdgeEmbedding(source_pos, dest_pos, self.device, D = self.edge_embd_dim // (2 * edge_pos_dim))
        self.geometry_encoder = GeometryEncoder2D(node_embd_dim=self.node_embd_dim)
        self.agent_cond_agent_edge_emb = nn.Embedding(1,self.edge_embd_dim, device=self.device)

        # components
        self.rho = RhoNN(num_node_feature=self.node_embd_dim, output_size = self.hidden_dim, num_edge_feature=self.edge_embd_dim, device=self.device)
        self.phi = PhiNN(num_node_feature=self.hidden_dim, output_size = self.hidden_dim, num_edge_feature=self.edge_embd_dim, device=self.device)
        self.psi = PsiNN(num_node_feature=self.hidden_dim, output_size = self.hidden_dim, num_edge_feature=self.edge_embd_dim, device=self.device)

    # εθ(Gk) = ψ(G(σ(Ga_l),ϕ(G_c(σ(Gt_l),{σ(G1:L_l )}1:N)))
    # might be going abuot it the wrong way. bottom up > top bottom
    # TODO: SA here
    def _process_observation_to_tensor(self, obs):
        point_clouds = torch.tensor(obs['point-clouds'], device = self.device, dtype=torch.float32)
        coords = torch.tensor(obs['coords'], device = self.device)
        agent_pos = torch.tensor(obs['agent-pos'], device = self.device, dtype=torch.float32)
        agent_state = obs['agent-state']
        agent_orientation = torch.tensor(obs['agent-orientation'], device = self.device, dtype=torch.float32)
        done = torch.tensor(obs['done'], device = self.device)
        time = torch.tensor(obs['time'], device = self.device)
        return point_clouds, coords, agent_pos, agent_state, agent_orientation, done, time

    def _get_selected_pointclouds(self, point_clouds, coords, centroids):
        matches = (coords[:, None, :] == centroids[None, :, :])  # [N, K, D]
        matches = matches.all(dim=2)  # [N, K]
        mask = matches.any(dim=1)     # [N] -> True where curr_coords matches a centroid
        selected_pointclouds = point_clouds[mask]
        return selected_pointclouds

    def _embed_local_graph(self,graph, features):

        # Embed Nodes first
        node_features_dict = dict()
        node_index_dict_by_type = dict()
        nodes, node_idx_dict_by_node = graph.get_nodes()


        for node in nodes:
            index = node_idx_dict_by_node[node]
        
            # x is used for agent state embedding
            x = index 
            if graph.timestep > self.pred_horizon:
                x += (graph.timestep - self.pred_horizon) * len(self.num)
            x = torch.tensor([x], device=self.device, dtype=torch.long)
            node_embd = None
            if node.type is NodeType.AGENT:
                agent_state = torch.tensor([graph.agent_state.value],device = self.device).float()
                agent_state_emb = self.agent_state_embedder(agent_state)
                agent_node_emb = self.agent_embedder(x).view(-1)
                node_embd = torch.cat([agent_node_emb, agent_state_emb], dim = -1)

            else:
                index -= self.num_agent_nodes # aget nodes always first nc implementation
                node_embd = features[index]


            
            if node.type not in node_index_dict_by_type.keys():
                node_index_dict_by_type[node.type] = [index]
                node_features_dict[node.type] = node_embd.view(1,-1)

            else:
                node_index_dict_by_type[node.type].append(index)
                node_features_dict[node.type] = torch.cat([node_features_dict[node.type], node_embd.view(1,-1)], dim = 0)

        

        # embed edges 
        edge_features_dict = dict()
        edge_index_dict = dict()
        connection_matrix = np.zeros((len(nodes), len(nodes)))
        edges = graph.get_edges()
        for edge in edges:
            source_node_idx = node_idx_dict_by_node[edge.source]
            dest_node_idx = node_idx_dict_by_node[edge.dest]
            connection_matrix[source_node_idx, dest_node_idx] = 1
            edge_emb = None 
            if edge.type is EdgeType.AGENT_COND_AGENT:
                edge_emb = self.agent_cond_agent_edge_emb(0) # think about this
            else:
                source_pos = edge.source.pos 
                dest_pos = edge.dest.pos
                edge_emb = self.spatial_edge_embedding(source_pos, dest_pos)

            if edge.type not in edge_index_dict.keys():
                edge_features_dict[edge.type]  = edge_emb.view(1,-1)
                edge_index_dict[edge.type] = [(source_node_idx,dest_node_idx)]
            else:
                edge_features_dict[edge.type]= torch.cat([edge_features_dict[edge.type], edge_emb.view(1,-1)], dim=0)
                edge_index_dict[edge.type].append((source_node_idx,dest_node_idx))
                
        for edge_type in edge_features_dict.keys():
            edge_features_dict[edge_type] = torch.tensor(edge_features_dict[edge_type], device = self.device)

        return node_features_dict, node_index_dict_by_type, edge_features_dict, edge_index_dict, connection_matrix

    def _make_localgraph(self,point_clouds, coords, agent_pos, agent_state, agent_orientation, timestep):
        # need to process point_clouds
        color_segments = defaultdict(list)

        
        for coord, color in zip(coords, point_clouds):
            # Convert RGB to a hashable tuple for grouping
            # color_key = tuple(color.astype(int))
            color_key = None
            color = tuple(color)
            if color == BLACK or color == YELLOW:
                color_key = 'goal'
            elif color == GREEN:
                color_key = 'edible'
            elif color == RED:
                color_key = 'obstacle'
            else:
                continue
            color_segments[color_key].append({
                'coord': coord,
                'color': color
            })        

        graph = LocalGraph(color_segments, timestep=timestep, agent_pos=agent_pos, agent_state=agent_state, agent_orientation=agent_orientation)
        return graph 

    def forward(self, 
                curr_obs, # 1 
                provided_demos, # list of demos, whereby each demo contains a list of observation,
                actions, # list of actions?
                ):
        

        # pass obs t
        pointclouds, coords, agent_pos, agent_state, agent_orientation, done, time = self._process_observation_to_tensor(curr_obs)

        curr_features, curr_centroids = self.geometry_encoder(coords)
        selected_pointclouds = self._get_selected_pointclouds(pointclouds, coords, curr_centroids)
        
        curr_graph_agent_node_embeddings_dict = dict()
        # # σ(Gt_l)
        # need to pass thru SA layers before nodes can be used to make local graph 
        curr_graph = self._make_localgraph(selected_pointclouds, curr_centroids, agent_pos, agent_state, agent_orientation, time)
        
        # embed current nodes and transform 
        curr_graph_X, curr_graph_node_idx_dict_by_type, curr_graph_E, edge_idx_dict, curr_graph_A = self._embed_local_graph(curr_graph, curr_features)
        
        rho_current = self.rho(curr_graph_X, curr_graph_node_idx_dict_by_type, curr_graph_A, curr_graph_E, edge_idx_dict)
        # store current agent node embeddings
        curr_agent_nodes_embeddings = rho_current[NodeType.AGENT]
        for agent_nodes,curr_agent_nodes_embedding in zip(curr_graph.agent_nodes, curr_agent_nodes_embeddings):
            curr_graph_agent_node_embeddings_dict[agent_nodes] = curr_agent_nodes_embedding


        # {σ(G1:L_l )}1:N
        demo_graphs = []
        demo_graph_agent_node_embeddings_dict = dict()
        for demo in provided_demos:
            graph_seq = []
            for obs in demo:
                pointclouds, coords, agent_pos, agent_state, agent_orientation, done, time = self._process_observation_to_tensor(obs)
                features, centroids = self.geometry_encoder(coords)
                selected_pointclouds = self._get_selected_pointclouds(pointclouds, coords, centroids)
                g = self._make_localgraph(selected_pointclouds, centroids, agent_pos, agent_state, agent_orientation, time)
                g_X, node_idx_dict_by_type, g_E, edge_idx_dict, g_A = self._embed_local_graph(g, features)
                _rho = self.rho(g_X, node_idx_dict_by_type, g_A, g_E, edge_idx_dict)
                agent_nodes_embeddings = _rho[NodeType.AGENT]
                for agent_node,agent_nodes_embedding in zip(g.agent_nodes, agent_nodes_embeddings):
                    demo_graph_agent_node_embeddings_dict[agent_node] = agent_nodes_embedding
                graph_seq.append(g)

            demo_graph = DemoGraph(graph_seq)
            demo_graphs.append(demo_graph)
        

        # G_c(σ(Gt_l),{σ(G1:L_l )}1:N)
        context_graph = ContextGraph(current_graph=curr_graph, demo_graphs=demo_graphs)

        # embed context graph nodes
        context_graph_nodes, context_graph_node_idx_dict_by_node = context_graph.get_temporal_nodes()
        context_graph_node_features = dict()
        context_graph_node_index_dict_by_type = dict()
        # hmm tensorfy this 
        # use rho_current and rho_demos to 'construct' new node features for self.phi
        for node in context_graph_nodes:
            index = context_graph_node_idx_dict_by_node[node]
            if node in demo_graph_agent_node_embeddings_dict.keys():
                features = demo_graph_agent_node_embeddings_dict[node]
            if node in curr_graph_agent_node_embeddings_dict.keys():
                features = curr_graph_agent_node_embeddings_dict[node]

            if node.type not in context_graph_node_features.keys():
                context_graph_node_index_dict_by_type[node.type] = [index]
                context_graph_node_features[node.type] = features.view(1,-1)
            else:
                context_graph_node_index_dict_by_type[node.type].append(index)
                context_graph_node_features[node.type] = torch.cat([context_graph_node_features[node.type],features.view(1,-1)])

        # then for edges
        context_graph_edges = context_graph.get_temporal_edges()
        context_graph_edge_features = dict()
        context_graph_edge_index_dict = dict()
        context_connection_matrix = np.zeros((len(context_graph_nodes), len(context_graph_nodes)))
        for edge in context_graph_edges:
            source_node_idx = context_graph_node_idx_dict_by_node[edge.source]
            dest_node_idx = context_graph_node_idx_dict_by_node[edge.dest]
            context_connection_matrix[source_node_idx, dest_node_idx] = 1
            feature = self.spatial_edge_embedding(edge.source.pos, edge.dest.pos)
            if edge.type not in context_graph_edge_features.keys():
                context_graph_edge_features[edge.type] = feature.view(1,-1)
                context_graph_edge_index_dict[edge.type] = [(source_node_idx,dest_node_idx)]

            else:
                context_graph_edge_features[edge.type] = torch.cat([context_graph_edge_features[edge.type], feature.view(1,-1)])
                context_graph_edge_index_dict[edge.type].append((source_node_idx,dest_node_idx))


        
        # ϕ(G_c(σ(Gt_l),{σ(G1:L_l )}1:N))
        phi = self.phi(context_graph_node_features, 
                       context_graph_node_index_dict_by_type, 
                       context_connection_matrix, 
                       context_graph_edge_features, 
                       context_graph_edge_index_dict)
        

        # use phi to update node features for current graph agent nodes 
        phi_agent_node_emb = phi[NodeType.AGENT]
        for curr_agent_node in curr_graph_agent_node_embeddings_dict.keys():
            node_idx = context_graph_node_idx_dict_by_node[curr_agent_node]
            type_idx = curr_graph_node_idx_dict_by_type[NodeType.AGENT].index(node_idx)
            curr_graph_agent_node_embeddings_dict[curr_agent_node] = phi_agent_node_emb[type_idx]


        # here will be pretty complex but whats happening is :
        # first construct action graph with curr graph 
        # then get embeddings for predicted graph with self.Rho
        # 
        # reconstruct node embedding for current_graph.agent_nodes, first with phi
        # then use curr_node_emb and actions to construct predictions
        predictions = torch.zeros((actions.shape[0], self.num_agent_nodes, self.hidden_dim),device = self.device)
        t = 0
        acc_action = None 
        for action in actions:
            
            action_obj = self._recover_action_obj(action)
            action_graph = ActionGraph(curr_graph, action_obj)

            predicted_graph = action_graph.predicted_graph

            #  since action graph made from current graph, we will use the same object features
            features = curr_features

            predicted_graph_X, predicted_graph_node_idx_dict_by_type, predicted_graph_E, edge_idx_dict, predicted_graph_A = self._embed_local_graph(predicted_graph, curr_features)
            rho_action = self.rho(predicted_graph_X, predicted_graph_node_idx_dict_by_type, predicted_graph_A, predicted_graph_E, edge_idx_dict)
            

            # then use rho_action and curr_node emb to update features of action node embeddings 
            action_nodes, action_index_dict_by_nodes = action_graph.get_action_nodes()
            action_node_embd = dict()
            action_node_idx_dict_by_type = dict()

            for action_node in action_nodes:
                node_index = action_index_dict_by_nodes[action_node]
                features = None
                # set features
                if action_node in curr_graph_agent_node_embeddings_dict.keys():
                    # get features from current node embeddings, ones that have agg. context
                    features = curr_graph_agent_node_embeddings_dict[action_node]
                else:
                    # get features from rho node embeddings, ones that have agg. future scene 
                    _index_offset = -4 # since 4 agent nodes are used, offset here 
                    type_index = predicted_graph_node_idx_dict_by_type[NodeType.AGENT].index(node_index + _index_offset)
                    features = rho_action[NodeType.AGENT][type_index]

                if node.type not in action_node_embd.keys():
                    action_node_idx_dict_by_type[node.type] = [node_index]
                    action_node_embd[node.type] =  features.view(1,-1) 
                else:
                    action_node_idx_dict_by_type[node.type].append(node_index)
                    action_node_embd[node.type] = torch.cat([action_node_embd[node.type],features.view(1,-1)])

            # then build edge emb for action graph
            action_edges = action_graph.get_action_edges()
            action_graph_edge_features = dict()
            action_graph_edge_index_dict = dict()
            action_connection_matrix = np.zeros((len(context_graph_nodes), len(context_graph_nodes)))
            for edge in action_edges:
                source_node_idx = action_index_dict_by_nodes[edge.source]
                dest_node_idx = action_index_dict_by_nodes[edge.dest]
                action_connection_matrix[source_node_idx, dest_node_idx] = 1
                feature = self.spatial_edge_embedding(edge.source.pos, edge.dest.pos)
                if edge.type not in action_graph_edge_features.keys():
                    action_graph_edge_features[edge.type] = feature.view(1,-1)
                    action_graph_edge_index_dict[edge.type] = [(source_node_idx,dest_node_idx)]

                else:
                    action_graph_edge_features[edge.type] = torch.cat([action_graph_edge_features[edge.type], feature.view(1,-1)])
                    action_graph_edge_index_dict[edge.type].append((source_node_idx,dest_node_idx))

            # finally get psi
            psi = self.psi(action_node_embd, action_node_idx_dict_by_type, action_connection_matrix, action_graph_edge_features, action_graph_edge_index_dict)



            # im going to move this part to InstantPolicyAgent
            # for now predictions will be node embds
            # ######################################################
            # simplify this since agent nodes need not move by same amount!
            # for now we just use center node, which is the last node

            agent_nodes = psi[NodeType.AGENT]
            predictions[t] = psi[NodeType.AGENT][self.num_agent_nodes:]

            t +=1
            # ######################################################

        return predictions

    def _recover_action_obj(self, action):
        if action.device == 'cpu':
            x, y, theta, state_change = action
        else:
            x, y, theta, state_change = action.cpu()
        forward_movement = np.sqrt(x**2 + y**2)
        
        # Method 2: Signed projection (can be negative if moving backward)
        forward_dir = np.array([np.cos(theta), np.sin(theta)])
        movement = np.array([x, y])
        forward_movement = np.dot(movement, forward_dir)

        # then get state change
        state_change = int(state_change)
        # breakpoint()
        return Action(forward_movement=forward_movement, rotation=theta, state_change=state_change)

## Modules/Networks for agent
# operates on local subgraphs G_l and propagates initial information about the point cloud observations to the gripper nodes
#  3 edge types : AGENT_TO_AGENT, AGENT_TO_OBJECT, OBJECT_TO_OBJECT
#  weights will be a num_nodes x num_nodes x num_features 
# 'nodes' will be a num_nodes x num_features matrix 
class RhoNN(nn.Module):

    def __init__(self, num_node_feature, output_size, num_edge_feature,device):
        super(RhoNN, self).__init__()
        self.device = device
        self.edge_types = [EdgeType.AGENT_TO_AGENT, EdgeType.OBJECT_TO_AGENT, EdgeType.OBJECT_TO_OBJECT]
        self.node_types = [node_type for node_type in NodeType]
        
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature,
                                       device=self.device
                                       )
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )

        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=output_size,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature,
                                       device=self.device
                                       )
        self.ln2 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )

    def forward(self,
                X : Dict[NodeType, torch.Tensor], # dictionary of matrix of node-features N x F 
                node_index_dict : Dict[NodeType, List[int]], # dictionary of list containing node-indexes
                A : torch.Tensor, # Adj matrix 
                E : Dict[EdgeType, torch.Tensor], # dictionary of matrix containing edge-features 
                edge_index_dict : Dict[EdgeType, List[Tuple[int,int]]], # dictionary containing list of tuple that corr to Connectivity matrix)
    ):
        z1 = self.l1(X, node_index_dict, A, E, edge_index_dict)
        z1 = self.ln1(X,z1)
        z2 = self.l2(z1, node_index_dict, A, E, edge_index_dict)
        z2 = self.ln2(z1,z2)

        return z2


# additionally propagates information through the demonstrated trajectories and allows all the relvant information from the context to be gathered at the gripper nodes of the current subgraph 
# 2 edgetypes involved : AGENT_COND_AGENT (demo to curr), AGENT_DEMO_AGENT (demo to demo)
#  weights will be a num_agent_nodes x num_agent_nodes x num_features 
# 'nodes' will be a num_agent_nodes x num_features matrix 
class PhiNN(nn.Module):

    def __init__(self, num_node_feature, output_size, num_edge_feature, device):
        super(PhiNN, self).__init__()
        self.device = device

        self.edge_types = [EdgeType.AGENT_COND_AGENT, EdgeType.AGENT_DEMO_AGENT]
        self.node_types = [node_type for node_type in NodeType]
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature,
                                       device=self.device)
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )
        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=output_size,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature,
                                       device=self.device)
        
        self.ln2 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )

    def forward(self,
                X : Dict[NodeType, torch.Tensor], # dictionary of matrix of node-features N x F 
                node_index_dict : Dict[NodeType, List[int]], # dictionary of list containing node-indexes
                A : torch.Tensor, # Adj matrix 
                E : Dict[EdgeType, torch.Tensor], # dictionary of matrix containing edge-features 
                edge_index_dict : Dict[EdgeType, List[Tuple[int,int]]], # dictionary containing list of tuple that corr to Connectivity matrix)
    ):
        z1 = self.l1(X, node_index_dict, A, E, edge_index_dict)
        z1 = self.ln1(X,z1)
        z2 = self.l2(z1, node_index_dict, A, E, edge_index_dict)
        z2 = self.ln2(z1,z2)

        return z2

# propagates information to nodes in the graph representing the actions
# 2 edgetypes involved : AGENT_TIME_ACTION_AGENT (curr graph to predicted action graph), 
# AGENT_DEMOACTION_AGENT (connect final frame of demo to predicted action graph)? (not used for now)
#  weights will be a num_agent_nodes x num_agent_nodes x num_features 
# 'nodes' will be a num_agent_nodes x num_features matrix 
class PsiNN(nn.Module):

    def __init__(self, num_node_feature, output_size, num_edge_feature,device):
        super(PsiNN, self).__init__()
        self.device = device

        self.edge_types = [EdgeType.AGENT_TIME_ACTION_AGENT]
        self.node_types = [node_type for node_type in NodeType]
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature,
                                       device=self.device)
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )

        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=output_size,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature,
                                       device=self.device)

        self.ln2 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )

    def forward(self,
                X : Dict[NodeType, torch.Tensor], # dictionary of matrix of node-features N x F 
                node_index_dict : Dict[NodeType, List[int]], # dictionary of list containing node-indexes
                A : torch.Tensor, # Adj matrix 
                E : Dict[EdgeType, torch.Tensor], # dictionary of matrix containing edge-features 
                edge_index_dict : Dict[EdgeType, List[Tuple[int,int]]], # dictionary containing list of tuple that corr to Connectivity matrix)
    ):
        z1 = self.l1(X, node_index_dict, A, E, edge_index_dict)
        z1 = self.ln1(X, z1)
        z2 = self.l2(z1, node_index_dict, A, E, edge_index_dict)
        z2 = self.ln1(z1, z2)
        
        return z2



# think about this hmm
class ChiNN(nn.Module):

    def __init__(self, num_node_feature, output_size, num_edge_feature):
        super(ChiNN, self).__init__()
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature)
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types
        )

        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=output_size,
                                       hidden_dim=output_size,
                                       num_edge_feature=num_edge_feature)

        self.ln2 = ResidualBlock(
            size = output_size,
            node_type=self.node_types
        )


# Aux building blocks
class ResidualBlock(nn.Module):

    def __init__(self, size, node_type, device):
        # Layer normalization for residuals
        super(ResidualBlock, self).__init__()
        self.device = device

        self.node_types = node_type
        self.layer_norms = nn.ModuleDict({
            node_type.name : nn.LayerNorm(size, device=self.device) for node_type in self.node_types
        })


    def forward(self, X, z):
        for node_type in z.keys():
            if node_type.name in self.layer_norms:
                # Layer norm
                normed = self.layer_norms[node_type.name](z[node_type])
                # Residual connection (only if dimensions match)
                if node_type in X and X[node_type].shape == normed.shape:
                    z[node_type] = normed + X[node_type]
                else:
                    z[node_type] = normed
        return z

# This transformer will act on 3 types of graph:
class HeteroAttentionLayer(nn.Module):

    # hidden dim in original paper is used bc of the occupancy net?
    # to overcome this, I will add an pre-linear layer to transform each node types to hidden dim
    """
    Args:
        num_node_feature is a dictionary with key as Nodetype, and value as num feature for that node type
        num_edge_feature is a dictionary with key as Edgetype, and value as num feature for that edge type
    """
    def __init__(self, 
                 node_types : List[NodeType], 
                 edge_types : List[EdgeType], 
                 num_node_features : int, 
                 num_edge_feature : int, 
                 device,
                 hidden_dim = 64, 
                 num_heads = 4, 
                 head_dim = 16):
        super(HeteroAttentionLayer, self).__init__()

        # network configurations 
        self.node_types = node_types
        self.output_dim = hidden_dim
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device = device
        self.head_dim = torch.tensor(head_dim)

        # functions declarations 
        self.softmax = torch.nn.Softmax(dim=-1)

        self.W1 = nn.ModuleDict({
            node_type.name : nn.Linear(num_node_features, hidden_dim, device=self.device)
            for node_type in node_types 
        })

        self.W2 = nn.ModuleDict({
            node_type.name : nn.Linear(num_node_features, hidden_dim, device=self.device)
            for node_type in node_types
        })

        self.W3 = nn.ModuleDict({
            node_type.name : nn.Linear(num_node_features, hidden_dim, device=self.device)
            for node_type in node_types
        })

        self.W4 = nn.ModuleDict({
            node_type.name : nn.Linear(num_node_features, hidden_dim, device=self.device)
            for node_type in node_types
        })

        self.W5 = nn.ModuleDict({
            edge_type.name : nn.Linear(num_edge_feature, hidden_dim, device=self.device)
            for edge_type in edge_types
        })
    # now need to reshape the matrixes into [self.num_heads, self.head_dim]
   
    def forward(self,
                X : Dict[NodeType, torch.Tensor], # dictionary of matrix of node-features N x F 
                node_index_dict : Dict[NodeType, List[int]], # dictionary of list containing node-indexes
                A : torch.Tensor, # Adj matrix 
                E : Dict[EdgeType, torch.Tensor], # dictionary of matrix containing edge-features 
                edge_index_dict : Dict[EdgeType, List[Tuple[int,int]]], # dictionary containing list of tuple that corr to Connectivity matrix
                ) -> Dict[NodeType, torch.Tensor]: 
        
        # first calculate W1Fi and w3fi and w5eij
        w1f = dict()
        w2f = dict()
        w3f = dict()
        w4f = dict()
        w5f = dict()

        for node_type, node_feature_matrix in X.items():
            w1f[node_type] = self.W1[node_type.name](node_feature_matrix)
            w2f[node_type] = self.W2[node_type.name](node_feature_matrix)
            w3f[node_type] = self.W3[node_type.name](node_feature_matrix)
            w4f[node_type] = self.W4[node_type.name](node_feature_matrix)

        for edge_type, edge_feature_matrix in E.items():
            w5f[edge_type] = self.W5[edge_type.name](edge_feature_matrix)
        # now to combine them 

        # first add w1f 
        final_X = w1f

    
        # TODO: maybe can parrellelise this
        # then iterate thru neighbours with adj matrix to get attention score and neigh(i) agg for nodes and edges
        # from adj matrix, we know which node and edge to select 
        for node_type, node_feature_matrix in w1f.items():
            for i in range(node_feature_matrix.shape[0]):
                curr_node_index = self._find_index(node_index_dict,i) # used to find nodes in node dictionary

                target_node_indexes = []
                summation = torch.zeros_like(final_X[curr_node_index[0]][curr_node_index[1]])
                for j in range(node_feature_matrix.shape[0]):
                    if A[i,j] != 0:
                        target_node_index = self._find_index(node_index_dict, j)
                        target_edge_index = self._find_index(edge_index_dict, (i,j))
                

                        _w3f = self._find_from_index(w3f,curr_node_index) # curr node 
                        _w4f = self._find_from_index(w4f,target_node_index) # adj node
                        _w5f = self._find_from_index(w5f,target_edge_index) # adj edge
                        _numerator = torch.matmul(_w3f.T, _w4f + _w5f)
                        _denominator = torch.sqrt(self.head_dim) 
                        attention_score = self.softmax(_numerator / _denominator)
                        _w2f = self._find_from_index(w2f, target_node_index)
                        summation += attention_score * (_w2f + _w5f)
                
                final_X[curr_node_index[0]][curr_node_index[1]] += summation                

        return final_X
    
    def _find_index(self, index_dict, index):
        for _type, indexes in index_dict.items():
            if index in indexes:
                target_index = indexes.index(index)
                return (_type, target_index)
    
    def _find_from_index(self,dicitonary, index):
        _type = index[0]
        _index = index[1]
        return dicitonary[_type][_index]




 
