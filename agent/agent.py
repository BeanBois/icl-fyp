import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np
from enum import Enum




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




"""

## Modules/Networks for agent
# operates on local subgraphs G_l and propagates initial information about the point cloud observations to the gripper nodes
class RhoNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RhoNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)


# additionally propagates information through the demonstrated trajectories and allows all the relvant information from the context to be gathered at the gripper nodes of the current subgraph 
class PhiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PhiNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

# propagates information to nodes in the graph representing the actions
class PsiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PsiNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

## Graph classes to represent objects. Consist of all classes that makes up classes 

# Classical GCN layer
class GCNLayer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).
    """
    def __init__(self, input_dim, output_dim, use_nonlinearity=True):
        super(GCNLayer, self).__init__()
        self.use_nonlinearity = use_nonlinearity
        self.Omega = nn.Parameter(torch.randn(input_dim, output_dim) * torch.sqrt(torch.tensor(2.0) / (input_dim + output_dim)))
        self.beta = nn.Parameter(torch.zeros(output_dim))

    def forward(self, H_k, A_normalized):
        agg = torch.matmul(A_normalized, H_k) # local agg
        H_k_next = torch.matmul(agg, self.Omega) + self.beta
        return F.relu(H_k_next) if self.use_nonlinearity else H_k_next


class GraphAttentionNetwork(nn.Module):
    def __init__(self,input_dim, dropout_rate, hidden_dim=2, output_dim=1):
        super(GraphAttentionNetwork,self).__init__()
        
        self.emb_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.a = nn.Parameter(torch.zeros(size=(2*hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.fc1_Omega = nn.Parameter(torch.randn(hidden_dim, output_dim) * torch.sqrt(torch.tensor(2.0) / (hidden_dim + output_dim)))
        self.fc1_beta = nn.Parameter(torch.zeros(output_dim))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        # # Batch normalization layers
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X, A):
        """
        Forward pass of the heterogeneous GCN.
        
        Args:
            X1 (torch.Tensor): Input node (of type 1) features of shape [num_type1_nodes, input_dim_type1]
            X2 (torch.Tensor): Input node (of type 2) features of shape [num_type2_nodes, input_dim_type2]
            adj (torch.Tensor): Normalized adjacency matrix of shape [num_type1_nodes + num_type2_nodes, num_type1_nodes + num_type2_nodes]
            X1_index (list): Indexes of type 1 node in Adj matrix
            X2_index (list): Indexes of type 2 node in Adj matrix
            
                                      
        Returns:
            torch.Tensor: Node classifications
        """
        # Create masks for different node types
        
        # 1) Linear transformation of input features
        H1 = self.fc1(X)  
        # H1 = self.dropout(H1)
        
        # 2) Compute attention coefficients with learnable parameters
        N = H1.size()[0]
        # self.bn1(H1)
        
        # Create concatenated pairs of nodes for attention computation
        a_input = torch.cat([H1.repeat(1, N).view(N * N, -1), 
                             H1.repeat(N, 1)], dim=1).view(N, N, 2 * self.emb_dim)
        
        # Apply attention mechanism
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        
        # 3) Apply softmax and mask with adjacency matrix
        # Set attention to zero for non-existing edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        
        # dropout attention
        attention = self.dropout(attention)
        
        # Apply attention to node features
        H2 = torch.matmul(attention, H1)
        
        # Output transformation
        output = torch.sigmoid(torch.matmul(H2, self.fc1_Omega) + self.fc1_beta)
        
        return output





# rethink how this is done 
# class Subgraph:

#     def __init__(self, tl, tr, br, bl, c):
#         self.nodes = [Node(tl),Node(tr),Node(br),Node(bl),Node(c)]
#         self.edges = self._init_edges()

#     # since we will be building the edge_matrix from the edges, we need to store create edge object with respective indexes
#     def _init_edge(self):





        
        
