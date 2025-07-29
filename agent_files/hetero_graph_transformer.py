import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from graph.aux import NodeType, EdgeType

class HeteroAttentionLayer(nn.Module):
    def __init__(self, 
                 node_types: List[NodeType], 
                 edge_types: List[EdgeType], 
                 num_heads: int = 16,     # Paper uses 16 heads
                 head_dim: int = 64):     # Paper uses 64 per head
        super().__init__()

        hidden_dim = num_heads * head_dim  # "hidden_dim must equal num_heads * head_dim"
        num_node_features = hidden_dim 
        num_edge_feature = num_node_features

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        

        # W1: Self-transformation (Equation 3)
        self.W1 = nn.ModuleDict({
            node_type.name: nn.Linear(num_node_features, hidden_dim)
            for node_type in node_types 
        })

        # W2: Value transformation 
        self.W2 = nn.ModuleDict({
            node_type.name: nn.Linear(num_node_features, hidden_dim)
            for node_type in node_types
        })

        # W3: Query transformation
        self.W3 = nn.ModuleDict({
            node_type.name: nn.Linear(num_node_features, hidden_dim)
            for node_type in node_types
        })

        # W4: Key transformation  
        self.W4 = nn.ModuleDict({
            node_type.name: nn.Linear(num_node_features, hidden_dim)
            for node_type in node_types
        })

        # W5: Edge transformation
        self.W5 = nn.ModuleDict({
            edge_type.name: nn.Linear(num_edge_feature, hidden_dim)
            for edge_type in edge_types
        })

    def forward(self,
                X: Dict[NodeType, torch.Tensor],
                node_index_dict: Dict[NodeType, List[int]], 
                A: torch.Tensor,
                E: Dict[EdgeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, List[Tuple[int,int]]]) -> Dict[NodeType, torch.Tensor]:
        
        # Step 1: Apply linear transformations
        w1f, w2f, w3f, w4f = {}, {}, {}, {}
        
        for node_type, features in X.items():
            w1f[node_type] = self.W1[node_type.name](features)  # Self-transformation
            w2f[node_type] = self.W2[node_type.name](features)  # Values
            w3f[node_type] = self.W3[node_type.name](features)  # Queries  
            w4f[node_type] = self.W4[node_type.name](features)  # Keys

        w5f = {}
        for edge_type, features in E.items():
            w5f[edge_type] = self.W5[edge_type.name](features)  # Edge features
            
        # Step 2: Reshape for multi-head attention
        for node_type in w1f.keys():
            batch_size = w1f[node_type].shape[0]
            w1f[node_type] = w1f[node_type].view(batch_size, self.num_heads, self.head_dim)
            w2f[node_type] = w2f[node_type].view(batch_size, self.num_heads, self.head_dim)
            w3f[node_type] = w3f[node_type].view(batch_size, self.num_heads, self.head_dim)
            w4f[node_type] = w4f[node_type].view(batch_size, self.num_heads, self.head_dim)

        for edge_type in w5f.keys():
            edge_batch_size = w5f[edge_type].shape[0] 
            w5f[edge_type] = w5f[edge_type].view(edge_batch_size, self.num_heads, self.head_dim)

        # Step 3: Compute attention (vectorized)
        output = {}
        
        for node_type in w1f.keys():
            num_nodes = w1f[node_type].shape[0]
            
            # Initialize output with self-transformation (W1 * Fi term)
            output[node_type] = w1f[node_type]  # [num_nodes, num_heads, head_dim]
            
            # Compute attention aggregation
            aggregated = torch.zeros_like(w1f[node_type])
            
            # For each node of this type
            for i in range(num_nodes):
                # Find this node's global index
                global_i = node_index_dict[node_type][i]
                
                # Find all neighbors
                neighbor_contributions = []
                attention_weights = []
                
                for j in range(A.shape[1]):  # All possible neighbors
                    if A[global_i, j] != 0:  # There's an edge
                        # Find neighbor's type and local index
                        neighbor_type, neighbor_local_idx = self._find_node_type(j, node_index_dict)
                        
                        # Find edge type and local index
                        edge_type, edge_local_idx = self._find_edge_type((global_i, j), edge_index_dict)
                        
                        if neighbor_type is not None and edge_type is not None:
                            # Get query, key, value, edge
                            query = w3f[node_type][i]  # [num_heads, head_dim]
                            key = w4f[neighbor_type][neighbor_local_idx]  # [num_heads, head_dim]
                            value = w2f[neighbor_type][neighbor_local_idx]  # [num_heads, head_dim] 
                            edge_feat = w5f[edge_type][edge_local_idx]  # [num_heads, head_dim]
                            
                            # Attention computation: att = softmax((W3*Fi)^T * (W4*Fj + W5*eij) / √d)
                            key_with_edge = key + edge_feat  # [num_heads, head_dim]
                            attention_scores = torch.sum(query * key_with_edge, dim=-1) / (self.head_dim ** 0.5)  # [num_heads]
                            
                            # Store for softmax normalization
                            attention_weights.append(attention_scores)
                            neighbor_contributions.append(value + edge_feat)
                
                if attention_weights:
                    # Normalize attention weights
                    attention_weights = torch.stack(attention_weights, dim=0)  # [num_neighbors, num_heads]
                    attention_weights = torch.softmax(attention_weights, dim=0)  # Softmax over neighbors
                    
                    # Aggregate neighbor contributions
                    neighbor_contributions = torch.stack(neighbor_contributions, dim=0)  # [num_neighbors, num_heads, head_dim]
                    
                    # Weighted sum
                    aggregated_contribution = torch.sum(
                        attention_weights.unsqueeze(-1) * neighbor_contributions, dim=0
                    )  # [num_heads, head_dim]
                    
                    aggregated[i] = aggregated_contribution

            # Add aggregated attention to self-transformation
            output[node_type] = output[node_type] + aggregated



        # Step 4: Reshape back to original format
        for node_type in output.keys():
            batch_size = output[node_type].shape[0]
            output[node_type] = output[node_type].view(batch_size, self.hidden_dim)

        return output

    def _find_node_type(self, global_idx: int, node_index_dict: Dict[NodeType, List[int]]) -> Tuple[NodeType, int]:
        """Find which node type and local index corresponds to global index"""
        for node_type, indices in node_index_dict.items():
            if global_idx in indices:
                local_idx = indices.index(global_idx)
                return node_type, local_idx
        return None, None

    def _find_edge_type(self, edge_tuple: Tuple[int, int], edge_index_dict: Dict[EdgeType, List[Tuple[int,int]]]) -> Tuple[EdgeType, int]:
        """Find which edge type and local index corresponds to edge tuple"""
        for edge_type, edge_list in edge_index_dict.items():
            if edge_tuple in edge_list:
                local_idx = edge_list.index(edge_tuple)
                return edge_type, local_idx
        return None, None