import torch
import torch.nn as nn

from agent.nn_components import HeterogeneousGraphTransformer

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
        # self.l1 = 


# additionally propagates information through the demonstrated trajectories and allows all the relvant information from the context to be gathered at the gripper nodes of the current subgraph 
class PhiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PhiNN, self).__init__()


# propagates information to nodes in the graph representing the actions
class PsiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PsiNN, self).__init__()




# rethink how this is done 
# class Subgraph:

#     def __init__(self, tl, tr, br, bl, c):
#         self.nodes = [Node(tl),Node(tr),Node(br),Node(bl),Node(c)]
#         self.edges = self._init_edges()

#     # since we will be building the edge_matrix from the edges, we need to store create edge object with respective indexes
#     def _init_edge(self):





        
        
