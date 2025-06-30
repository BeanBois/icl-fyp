import torch
import torch.nn as nn

from agent.nn_components import HeterogeneousGraphTransformer

import numpy as np
from enum import Enum

from ..utils.graph import EdgeType, NodeType
from typing import Dict, List, Tuple



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

    def __init__(self, num_node_feature, num_edge_feature):
        super(RhoNN, self).__init__()
        self.edge_types = [EdgeType.AGENT_TO_AGENT, EdgeType.AGENT_TO_OBJECT, EdgeType.OBJECT_TO_OBJECT]
        self.node_types = [node_type for node_type in NodeType]
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       num_edge_feature=num_edge_feature)
        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       num_edge_feature=num_edge_feature)

    def forward(self,
                X : Dict[NodeType, torch.Tensor], # dictionary of matrix of node-features N x F 
                node_index_dict : Dict[NodeType, List[int]], # dictionary of list containing node-indexes
                A : torch.Tensor, # Adj matrix 
                E : Dict[EdgeType, torch.Tensor], # dictionary of matrix containing edge-features 
                edge_index_dict : Dict[EdgeType, List[Tuple[int,int]]], # dictionary containing list of tuple that corr to Connectivity matrix)
    ):
        z1 = self.l1(X, node_index_dict, A, E, edge_index_dict)
        z2 = self.l2(z1, node_index_dict, A, E, edge_index_dict)

        return z2


# additionally propagates information through the demonstrated trajectories and allows all the relvant information from the context to be gathered at the gripper nodes of the current subgraph 
# 2 edgetypes involved : AGENT_COND_AGENT (demo to curr), AGENT_DEMO_AGENT (demo to demo)
#  weights will be a num_agent_nodes x num_agent_nodes x num_features 
# 'nodes' will be a num_agent_nodes x num_features matrix 
class PhiNN(nn.Module):

    def __init__(self, num_node_feature, num_edge_feature):
        super(PhiNN, self).__init__()
        self.edge_types = [EdgeType.AGENT_COND_AGENT, EdgeType.AGENT_DEMO_AGENT]
        self.node_types = [node_type for node_type in NodeType]
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       num_edge_feature=num_edge_feature)
        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       num_edge_feature=num_edge_feature)

    def forward(self,
                X : Dict[NodeType, torch.Tensor], # dictionary of matrix of node-features N x F 
                node_index_dict : Dict[NodeType, List[int]], # dictionary of list containing node-indexes
                A : torch.Tensor, # Adj matrix 
                E : Dict[EdgeType, torch.Tensor], # dictionary of matrix containing edge-features 
                edge_index_dict : Dict[EdgeType, List[Tuple[int,int]]], # dictionary containing list of tuple that corr to Connectivity matrix)
    ):
        z1 = self.l1(X, node_index_dict, A, E, edge_index_dict)
        z2 = self.l2(z1, node_index_dict, A, E, edge_index_dict)
        
        return z2


# propagates information to nodes in the graph representing the actions
# 2 edgetypes involved : AGENT_TIME_ACTION_AGENT (curr graph to predicted action graph), 
# AGENT_DEMOACTION_AGENT (connect final frame of demo to predicted action graph)? (not used for now)
#  weights will be a num_agent_nodes x num_agent_nodes x num_features 
# 'nodes' will be a num_agent_nodes x num_features matrix 
class PsiNN(nn.Module):

    def __init__(self, num_node_feature, num_edge_feature):
        super(PsiNN, self).__init__()
        self.edge_types = [EdgeType.AGENT_TIME_ACTION_AGENT]
        self.node_types = [node_type for node_type in NodeType]
        self.l1 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       num_edge_feature=num_edge_feature)
        self.l2 = HeteroAttentionLayer(node_types=self.node_types, 
                                       edge_types=self.edge_types,
                                       num_node_features=num_node_feature,
                                       num_edge_feature=num_edge_feature)



    def forward(self,
                X : Dict[NodeType, torch.Tensor], # dictionary of matrix of node-features N x F 
                node_index_dict : Dict[NodeType, List[int]], # dictionary of list containing node-indexes
                A : torch.Tensor, # Adj matrix 
                E : Dict[EdgeType, torch.Tensor], # dictionary of matrix containing edge-features 
                edge_index_dict : Dict[EdgeType, List[Tuple[int,int]]], # dictionary containing list of tuple that corr to Connectivity matrix)
    ):
        z1 = self.l1(X, node_index_dict, A, E, edge_index_dict)
        z2 = self.l2(z1, node_index_dict, A, E, edge_index_dict)
        
        return z2


# This transformer will act on 3 types of graph:
    # 
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
                 num_node_features : Dict[NodeType, int], 
                 num_edge_feature : Dict[EdgeType, int], 
                 hidden_dim = 1024, 
                 num_heads = 16, 
                 head_dim = 64):
        super(HeteroAttentionLayer, self).__init__()

        # network configurations 
        self.node_types = node_types
        self.output_dim = hidden_dim
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # functions declarations 
        self.softmax = torch.nn.Softmax

        self.W1 = nn.ModuleDict({
            node_type : nn.Linear(num_node_features[node_type], hidden_dim)
            for node_type in node_types 
        })

        self.W2 = nn.ModuleDict({
            node_type : nn.Linear(num_node_features[node_type], hidden_dim)
            for node_type in node_types
        })

        self.W3 = nn.ModuleDict({
            node_type : nn.Linear(num_node_features[node_type], hidden_dim)
            for node_type in node_types
        })

        self.W4 = nn.ModuleDict({
            node_type : nn.Linear(num_node_features[node_type], hidden_dim)
            for node_type in node_types
        })

        self.W5 = nn.ModuleDict({
            edge_type : nn.Linear(num_edge_feature[edge_type], hidden_dim)
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

        for node_type, node_feature_matrix in X:
            w1f[node_type] = self.W1[node_type](node_feature_matrix)
            w2f[node_type] = self.W2[node_type](node_feature_matrix)
            w3f[node_type] = self.W3[node_type](node_feature_matrix)
            w4f[node_type] = self.W4[node_type](node_feature_matrix)

        for edge_type, edge_feature_matrix in E:
            w5f[edge_type] = self.W5[edge_type](edge_feature_matrix)

        # now to combine them 

        # first add w1f 
        final_X = w1f

    
        # then iterate thru neighbours with adj matrix to get attention score and neigh(i) agg for nodes and edges
        # from adj matrix, we know which node and edge to select 
        for node_type, node_feature_matrix in w1f:
            for i in range(node_feature_matrix.size[0]):
                curr_node_index = self._find_index(node_index_dict,i) # used to find nodes in node dictionary

                target_node_indexes = []
                summation = torch.zeros_like(final_X[curr_node_index[0]][curr_node_index[1]])
                for j in range(node_feature_matrix.size[0]):
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
        for _type, indexes in index_dict:
            if index in indexes:
                target_index = indexes.index(index)
                return (_type, target_index)
    
    def _find_from_index(self,dicitonary, index):
        _type = index[0]
        _index = index[1]
        return dicitonary[_type][_index]









        
        
