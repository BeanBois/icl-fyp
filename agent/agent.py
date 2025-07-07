import torch
import torch.nn as nn

from agent.nn_components import HeterogeneousGraphTransformer

import numpy as np
from enum import Enum

# from ..utils.graph import DemoGraph, ContextGraph, ActionGraph, EdgeType, NodeType, make_localgraph # use when running this file
from utils.graph import DemoGraph, ContextGraph, ActionGraph, EdgeType, NodeType, make_localgraph # use when running from rwd

# from ..tasks.twoD.game import Action # use when running this file
from tasks.twoD.game import Action# use when running from rwd

from typing import Dict, List, Tuple


# We use layer normalisation layers (Ba, 2016) between every attention layer and add additional residual connections to 
# ensure good propagation of gradients throughout the network. 
# TODO apply layer norm and residuals to rho, psi phi
# Layer normalization
# self.layer_norms = nn.ModuleDict({
#     node_type: nn.ModuleList([
#         nn.LayerNorm(hidden_dim),
#         nn.LayerNorm(hidden_dim)
#     ]) for node_type in node_types
# })
# ...
# Apply layer norm and residual
# for node_type in output1.keys():
#     if node_type in combined_graph['node_features']:
#         normed = self.layer_norms[node_type][0](output1[node_type])
#         output1[node_type] = normed + combined_graph['node_features'][node_type]
        

# # Network to combine multiple demonstration trajectories
# self.demo_aggregation = nn.ModuleDict({
#     node_type: nn.Linear(hidden_dim, hidden_dim)
#     for node_type in node_types
# })

# Auxiliary functions/ classes 


# ARCHITECTURES 
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

# to account for node features, we do it at this level lmao solves 1,2,4 lol
# 3 is just tuff icl
# but still need to remove the node and edge features and all that sadge
# for agent node embd use nn.Embedding to embed num_agent_node * prediciton steps 
# flatten and change to tensor
# def SinCosEdgeEmbedding(source, dest, D = 3):

#     num_feature = source.shape[0]
#     embedding = np.zeros((num_feature, 2 * D))
#     diff = dest - source 
#     aux_func = lambda d : np.array([np.sin(2**d  * np.pi * diff), np.cos(2**d  * np.pi * diff)]) 

#     for d in range(D):
#         embedding[:,d:d+2] =  aux_func(d)
#     return embedding

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
        self.object_embedders = nn.ModuleDict({
            node_type.name : nn.Embedding(1,self.node_embd_dim, device=self.device)
            for node_type in NodeType if node_type != NodeType.AGENT
        }) # 1 embedder for each node type except for agent nodes since they share similar geometry
        self.agent_cond_agent_edge_emb = nn.Embedding(1,self.edge_embd_dim, device=self.device)

        # components
        self.rho = RhoNN(num_node_feature=self.node_embd_dim, output_size = self.hidden_dim, num_edge_feature=self.edge_embd_dim, device=self.device)
        self.phi = PhiNN(num_node_feature=self.hidden_dim, output_size = self.hidden_dim, num_edge_feature=self.edge_embd_dim, device=self.device)
        self.psi = PsiNN(num_node_feature=self.hidden_dim, output_size = self.hidden_dim, num_edge_feature=self.edge_embd_dim, device=self.device)

        self.chi = None 
        # an additional layer that processes the agent nodes after psi. main point is to agg with attention mechanism on how centre of mass should be adjusted
        # with the given features from the predicted graph 
        # topology of this graph will be customised for the object 
        # 

        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim, device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,2, device=self.device),
        )

        self.pred_head_rot = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim,device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,1,device=self.device),
        )

        self.pred_head_g = nn.Sequential(
            nn.Linear(self.hidden_dim, self.node_embd_dim,device=self.device),
            nn.GELU(),
            nn.Linear(self.node_embd_dim,1,device=self.device),
        )


    # εθ(Gk) = ψ(G(σ(Ga_l),ϕ(G_c(σ(Gt_l),{σ(G1:L_l )}1:N)))
    # might be going abuot it the wrong way. bottom up > top bottom
    def _embed_local_graph(self,graph):

        # Embed Nodes first
        node_features_dict = dict()
        node_index_dict_by_type = dict()
        nodes, node_idx_dict_by_node = graph.get_nodes()


        for node in nodes:
            index = node_idx_dict_by_node[node]
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
                embedder = self.object_embedders[node.type.name] # think about this too
                node_embd = embedder(torch.tensor([0],device = self.device, dtype=torch.long))


            
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


    def forward(self, 
                curr_obs, # 1 
                provided_demos, # list of demos, whereby each demo contains a list of observation,
                actions, # list of actions?
                # agent_keypoints, # dict of keypoints str : vector
                ):
        curr_graph_agent_node_embeddings_dict = dict()
        # # σ(Gt_l)
        curr_graph = make_localgraph(curr_obs)
        # embed current nodes and transform
        curr_graph_X, curr_graph_node_idx_dict_by_type, curr_graph_E, edge_idx_dict, curr_graph_A = self._embed_local_graph(curr_graph)
        
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
                g = make_localgraph(obs)
                g_X, node_idx_dict_by_type, g_E, edge_idx_dict, g_A = self._embed_local_graph(g)
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
        predictions = torch.zeros((actions.shape),device = self.device)
        t = 0
        for action in actions:
            action_obj = self._recover_action_obj(action)
            action_graph = ActionGraph(curr_graph, action_obj)

            predicted_graph = action_graph.predicted_graph

            predicted_graph_X, predicted_graph_node_idx_dict_by_type, predicted_graph_E, edge_idx_dict, predicted_graph_A = self._embed_local_graph(predicted_graph)
            # get σ(Ga_l)
            # error here?
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



            # simplify this since agent nodes need not move by same amount!
            # for now we just use center node, which is the last node
            # agent_nodes = psi[NodeType.AGENT][4:, ...]
            agent_nodes = psi[NodeType.AGENT][-1, ...]
            agent_translation = self.pred_head(agent_nodes)
            agent_rotation = self.pred_head_rot(agent_nodes)
            agent_state_change = self.pred_head_g(agent_nodes)
            # need to further agg. since the predictions comes out for 8 diff nodes. 
            # breakpoint()

            predictions[t, :2] = agent_translation
            predictions[t, 2] = agent_rotation
            predictions[t, 3] = agent_state_change
            t +=1

        return predictions

    def _recover_action_obj(self, action):
        x, y, theta, state_change = action
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

class ResidualBlock(nn.Module):

    def __init__(self, size, node_type, device):
        # Layer normalization for residuals
        super(ResidualBlock, self).__init__()

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









        
        
