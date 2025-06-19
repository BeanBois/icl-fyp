import torch
import torch.nn as nn

from agent.nn_components import HeterogeneousGraphTransformer

import numpy as np
from enum import Enum

from ..utils.graph import EdgeType, NodeType



class InstantPolicyAgent:

    def __init__(self):
        pass

    def _train(self):
        pass

    def _eval(self):
        pass

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



## Modules/Networks for agent
# operates on local subgraphs G_l and propagates initial information about the point cloud observations to the gripper nodes
#  3 edge types : AGENT_TO_AGENT, AGENT_TO_OBJECT, OBJECT_TO_OBJECT
#  weights will be a num_nodes x num_nodes x num_features 
# 'nodes' will be a num_nodes x num_features matrix 
class RhoNN(nn.Module):

    def __init__(self, num_nodes, num_node_feature, num_edges, num_edge_feature):
        super(RhoNN, self).__init__()
        self.edge_types = [EdgeType.AGENT_TO_AGENT, EdgeType.AGENT_TO_OBJECT, EdgeType.OBJECT_TO_OBJECT]
        self.node_types = [node_type for node_type in NodeType]


# additionally propagates information through the demonstrated trajectories and allows all the relvant information from the context to be gathered at the gripper nodes of the current subgraph 
# 2 edgetypes involved : AGENT_COND_AGENT (demo to curr), AGENT_DEMO_AGENT (demo to demo)
#  weights will be a num_agent_nodes x num_agent_nodes x num_features 
# 'nodes' will be a num_agent_nodes x num_features matrix 
class PhiNN(nn.Module):

    def __init__(self, num_nodes, num_node_feature, num_edges, num_edge_feature):
        super(PhiNN, self).__init__()
        self.edge_types = [EdgeType.AGENT_COND_AGENT, EdgeType.AGENT_DEMO_AGENT]
        self.node_types = [node_type for node_type in NodeType]


# propagates information to nodes in the graph representing the actions
# 2 edgetypes involved : AGENT_TIME_ACTION_AGENT (curr graph to predicted action graph), 
# AGENT_DEMOACTION_AGENT (connect final frame of demo to predicted action graph)? (not used for now)
#  weights will be a num_agent_nodes x num_agent_nodes x num_features 
# 'nodes' will be a num_agent_nodes x num_features matrix 
class PsiNN(nn.Module):

    def __init__(self, num_nodes, num_node_feature, num_edges, num_edge_feature):
        super(PsiNN, self).__init__()
        self.edge_types = [EdgeType.AGENT_TIME_ACTION_AGENT]
        self.node_types = [node_type for node_type in NodeType]






# This transformer will act on 3 types of graph:
    # 
class HeteroAttentionLayer(nn.Module):

    # hidden dim in original paper is used bc of the occupancy net?
    # to overcome this, I will add an pre-linear layer to transform each node types to hidden dim
    def __init__(self, node_types, edge_types, hidden_dim = 1024, num_heads = 16, head_dim = 64):
        super(HeteroAttentionLayer, self).__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # functions declarations 
        self.softmax = torch.nn.Softmax
        


        self.W1 = nn.ModuleDict({
            node_type : nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types 
        })

        self.W2 = nn.ModuleDict({
            node_type : nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types
        })

        self.W3 = nn.ModuleDict({
            node_type : nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types
        })

        self.W4 = nn.ModuleDict({
            node_type : nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types
        })

        self.W5 = nn.ModuleDict({
            edge_type : nn.Linear(hidden_dim, hidden_dim)
            for edge_type in edge_types
        })


    def _attention_mechanism(self,X,A):
        pass 

    # Here X holds num_nodes * features_size (hidden_dim)
    def forward(self,X,A): 
        # first apply self W1Fi

        pass







        
        
