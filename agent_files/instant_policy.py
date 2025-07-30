# contains psi, rho ... 
from .hetero_graph_transformer import HeteroAttentionLayer
from graph import EdgeType, NodeType, ActionGraph, LocalGraph, DemoGraph, ContextGraph
from .geometry_encoder import GeometryEncoder2D
from tasks2d import RED, GREEN, BLACK, YELLOW, WHITE, PURPLE, BLUE
from tasks2d import LousyPacmanPseudoGameAction as Action 
from tasks2d import LousyPacmanPlayerState as PlayerState





import torch 
import torch.nn as nn 

import numpy as np 

from typing import List,Dict,Tuple
from collections import defaultdict 


# TODO: add Pointnet++ and SA layer! for scene node embedding
# def SinCosEdgeEmbedding(source, dest, device, D=3):
#     num_feature = source.shape[0]
#     embedding = torch.zeros((num_feature, 2 * D), device=device)
#     diff = torch.tensor(dest - source)
#     for d in range(D):
#         sin_vals = torch.sin(2**d * torch.pi * diff)
#         cos_vals = torch.cos(2**d * torch.pi * diff)
#         embedding[:, 2*d] = sin_vals
#         embedding[:, 2*d+1] = cos_vals
    
#     return embedding

def SinCosEdgeEmbedding(source, dest, device, D=3):
    """
    Fixed version that produces EXACTLY the same output as original, just handles NaN
    """
    num_feature = source.shape[0]
    embedding = torch.zeros((num_feature, 2 * D), device=device)
    
    # Calculate diff - but add NaN protection
    diff = dest - source
    
    # Handle NaN/Inf in diff (replace with 0)
    if torch.isnan(diff).any() or torch.isinf(diff).any():
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    

    
    # Clamp extreme values to prevent sin/cos overflow
    diff = torch.clamp(diff, min=-1000, max=1000)
    
    for d in range(D):
        sin_vals = torch.sin(2**d * torch.pi * diff)
        cos_vals = torch.cos(2**d * torch.pi * diff)
        
        # Additional NaN check on sin/cos results
        sin_vals = torch.nan_to_num(sin_vals, nan=0.0)
        cos_vals = torch.nan_to_num(cos_vals, nan=0.0)
        
        embedding[:, 2*d] = sin_vals
        embedding[:, 2*d+1] = cos_vals
    
    return embedding


# add SA layer here
class InstantPolicy(nn.Module):
    
    def __init__(self, 
                device,
                num_agent_nodes = 4, 
                pred_horizon = 5, 
                num_att_heads = 16,
                head_dim = 64,
                agent_state_embd_dim = 64,
                edge_pos_dim = 2):
        super(InstantPolicy, self).__init__()


        hidden_dim = num_att_heads * head_dim
        node_embd_dim = hidden_dim
        edge_embd_dim = node_embd_dim

        self.device = device
        self.num_agent_nodes = num_agent_nodes
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.node_embd_dim = node_embd_dim
        self.edge_embd_dim = edge_embd_dim
        self.agent_state_embd_dim = agent_state_embd_dim
        self.edge_pos_dim = edge_pos_dim

        # embedders
        self.agent_embedder = nn.Embedding(
            self.num_agent_nodes * self.pred_horizon,
            self.node_embd_dim - self.agent_state_embd_dim, 
            device=self.device
        )
        self.agent_state_embedder = nn.Linear(1, self.agent_state_embd_dim, device=self.device)

        self.spatial_edge_embedding = SinCosEdgeEmbedding
        self.geometry_encoder = GeometryEncoder2D(node_embd_dim=self.node_embd_dim, device=self.device).to(self.device)
        self.agent_cond_agent_edge_emb = nn.Embedding(1,self.edge_embd_dim, device=self.device)

        # components
        self.rho = RhoNN(num_att_heads=num_att_heads, head_dim=head_dim, device=self.device)
        self.phi = PhiNN(num_att_heads=num_att_heads, head_dim=head_dim, device=self.device)
        self.psi = PsiNN(num_att_heads=num_att_heads, head_dim=head_dim, device=self.device)

    # εθ(Gk) = ψ(G(σ(Ga_l),ϕ(G_c(σ(Gt_l),{σ(G1:L_l )}1:N)))
    # might be going abuot it the wrong way. bottom up > top bottom
    # TODO: SA here
    def _process_observation_to_tensor(self, obs):
        point_clouds = torch.tensor(obs['point-clouds'], device = self.device, dtype=torch.float32)
        coords = torch.tensor(obs['coords'], device = self.device, dtype=torch.float32)
        agent_pos = torch.tensor(obs['agent-pos'], device = self.device, dtype=torch.float32)
        agent_state = obs['agent-state']
        agent_orientation = torch.tensor(obs['agent-orientation'], device = self.device, dtype=torch.float32) # angle in degree
        done = torch.tensor(obs['done'], device = self.device)
        time = torch.tensor(obs['time'], device = self.device)
        return point_clouds, coords, agent_pos, agent_state, agent_orientation, done, time

    def _get_selected_pointclouds(self, point_clouds, coords, centroids):
        matches = (coords[:, None, :] == centroids[None, :, :])  # [N, K, D]
        matches = matches.all(dim=2)  # [N, K]
        mask = matches.any(dim=1)     # [N] -> True where curr_coords matches a centroid
        selected_pointclouds = point_clouds[mask]
        return selected_pointclouds

    # yea something is wrong here 
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
                _index = index -  self.num_agent_nodes # aget nodes always first nc implementation
                node_embd = features[_index]


            
            if node.type not in node_index_dict_by_type.keys():
                node_index_dict_by_type[node.type] = [index]
                node_features_dict[node.type] = node_embd.view(1,-1)

            else:
                node_index_dict_by_type[node.type].append(index)
                node_features_dict[node.type] = torch.cat([node_features_dict[node.type], node_embd.view(1,-1)], dim = 0)

        

        # embed edges 
        edge_features_dict = dict()
        edge_index_dict = dict()
        connection_matrix = torch.zeros((len(nodes), len(nodes)), device=self.device)
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
                edge_emb = self.spatial_edge_embedding(source_pos, dest_pos, device= self.device, D = self.edge_embd_dim // (2 * self.edge_pos_dim))


            if edge.type not in edge_index_dict.keys():
                edge_features_dict[edge.type]  = edge_emb.view(1,-1)
                edge_index_dict[edge.type] = [(source_node_idx,dest_node_idx)]
            else:
                edge_features_dict[edge.type]= torch.cat([edge_features_dict[edge.type], edge_emb.view(1,-1)], dim=0)
                edge_index_dict[edge.type].append((source_node_idx,dest_node_idx))
                

        # already a tensor
        # for edge_type in edge_features_dict.keys():
            # edge_features_dict[edge_type] = torch.stack(edge_features_dict[edge_type])

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

    # TODO : fix the graphs here. it is notworking as it should? unsure im not going to lie
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
            feature = self.spatial_edge_embedding(edge.source.pos, edge.dest.pos, device= self.device, D = self.edge_embd_dim // (2 * self.edge_pos_dim))
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
                feature = self.spatial_edge_embedding(edge.source.pos, edge.dest.pos, device= self.device, D = self.edge_embd_dim // (2 * self.edge_pos_dim))
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

        x, y, theta, state_change = action
        forward_movement = np.sqrt(x**2 + y**2)
        
        # Method 2: Signed projection (can be negative if moving backward)
        forward_dir = np.array([np.cos(theta), np.sin(theta)])
        movement = np.array([x, y])
        forward_movement = np.dot(movement, forward_dir)

        # then get state change
        state_change = int(abs(state_change) > 0.5)
        theta_deg = torch.rad2deg(theta)

        # state_change is a
        for player_state in PlayerState:
            if state_change == player_state.value:
                state_change = player_state 
                continue

        assert type(state_change) == PlayerState

        return Action(forward_movement=forward_movement, rotation_deg=theta_deg, state_change=state_change)

## Modules/Networks for agent
# operates on local subgraphs G_l and propagates initial information about the point cloud observations to the gripper nodes
#  3 edge types : AGENT_TO_AGENT, AGENT_TO_OBJECT, OBJECT_TO_OBJECT
#  weights will be a num_nodes x num_nodes x num_features 
# 'nodes' will be a num_nodes x num_features matrix 
class RhoNN(nn.Module):

    def __init__(self, num_att_heads, head_dim,device):
        super(RhoNN, self).__init__()
        self.device = device
        self.edge_types = [EdgeType.AGENT_TO_AGENT, EdgeType.OBJECT_TO_AGENT, EdgeType.OBJECT_TO_OBJECT]
        self.node_types = [node_type for node_type in NodeType]
        output_size = num_att_heads * head_dim

        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )

        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)
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

    def __init__(self, num_att_heads, head_dim, device):
        super(PhiNN, self).__init__()
        self.device = device
        output_size = num_att_heads * head_dim

        self.edge_types = [EdgeType.AGENT_COND_AGENT, EdgeType.AGENT_DEMO_AGENT]
        self.node_types = [node_type for node_type in NodeType]
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )
        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)
        
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

    def __init__(self, num_att_heads, head_dim ,device):
        super(PsiNN, self).__init__()

        self.device = device
        output_size = num_att_heads * head_dim

        self.edge_types = [EdgeType.AGENT_TIME_ACTION_AGENT]
        self.node_types = [node_type for node_type in NodeType]
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types,
            device=self.device
        )

        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)

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
        z2 = self.ln2(z1, z2)
        
        return z2



# think about this hmm
class ChiNN(nn.Module):

    def __init__(self, num_att_heads, head_dim):
        output_size =  num_att_heads * head_dim
        super(ChiNN, self).__init__()

        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)
        
        self.ln1 = ResidualBlock(
            size = output_size,
            node_type=self.node_types
        )

        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_heads=num_att_heads, 
                                       head_dim=head_dim)

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
