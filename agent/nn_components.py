
# trying out another way 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from utils.graph import NodeType, EdgeType

class HeterogeneousGraphTransformer(nn.Module):
    
    def __init__(self, 
                 node_types: List[NodeType],
                 edge_types: List[EdgeType], 
                 hidden_dim: int = 1024,
                 num_heads: int = 16,
                 head_dim: int = 64):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Separate weight matrices for each node/edge type combination
        self.W1 = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types
        })
        
        self.W2 = nn.ModuleDict({
            f"{src_type}_{edge_type}_{dst_type}": nn.Linear(hidden_dim, hidden_dim)
            for src_type in node_types
            for dst_type in node_types  
            for edge_type in edge_types
        })
        
        self.W3 = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, num_heads * head_dim)
            for node_type in node_types
        })
        
        self.W4 = nn.ModuleDict({
            f"{src_type}_{edge_type}_{dst_type}": nn.Linear(hidden_dim, num_heads * head_dim)
            for src_type in node_types
            for dst_type in node_types
            for edge_type in edge_types
        })
        
        self.W5 = nn.ModuleDict({
            edge_type: nn.Linear(self._get_edge_dim(edge_type), num_heads * head_dim)
            for edge_type in edge_types
        })
        
        self.layer_norm = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })
        
    def _get_edge_dim(self, edge_type: str) -> int:
        """Get edge feature dimension based on edge type"""
        # For position-based edges with sinusoidal encoding
        if 'position' in edge_type:
            return 60  # 30 frequencies * 2 (sin, cos) for 3D positions
        else:
            return 32  # Generic edge features
    
    def forward(self, 
                node_features: Dict[str, torch.Tensor],
                edge_indices: Dict[str, torch.Tensor], 
                edge_features: Dict[str, torch.Tensor],
                node_types_per_node: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: Dict mapping node type to features [num_nodes, hidden_dim]
            edge_indices: Dict mapping edge type to [2, num_edges] 
            edge_features: Dict mapping edge type to [num_edges, edge_dim]
            node_types_per_node: [num_nodes] tensor indicating type of each node
        """
        
        # Concatenate all node features
        all_nodes = torch.cat([node_features[nt] for nt in self.node_types], dim=0)
        num_nodes = all_nodes.size(0)
        
        # Initialize output features
        updated_features = torch.zeros_like(all_nodes)
        
        # Self-transformation (W1 * F_i term)
        node_start_idx = 0
        for node_type in self.node_types:
            node_count = node_features[node_type].size(0)
            node_end_idx = node_start_idx + node_count
            
            # Apply W1 transformation for this node type
            updated_features[node_start_idx:node_end_idx] = \
                self.W1[node_type](all_nodes[node_start_idx:node_end_idx])
            
            node_start_idx = node_end_idx
        
        # Attention-based aggregation
        for edge_type, edge_idx in edge_indices.items():
            if edge_idx.size(1) == 0:  # No edges of this type
                continue
                
            src_nodes, dst_nodes = edge_idx[0], edge_idx[1]
            edge_feat = edge_features[edge_type]
            
            # Get node types for source and destination
            src_types = node_types_per_node[src_nodes] # returns the types of source nodes
            dst_types = node_types_per_node[dst_nodes]
            
            # Process edges by src_type, dst_type combination
            for src_type_idx, src_type in enumerate(self.node_types):
                for dst_type_idx, dst_type in enumerate(self.node_types):
                    
                    # Find edges of this type combination
                    mask = (src_types == src_type_idx) & (dst_types == dst_type_idx)
                    if not mask.any():
                        continue
                    
                    edge_key = f"{src_type}_{edge_type}_{dst_type}"
                    
                    # Get relevant edges and nodes
                    relevant_src = src_nodes[mask]
                    relevant_dst = dst_nodes[mask]
                    relevant_edges = edge_feat[mask]
                    
                    # Compute attention scores
                    src_features = all_nodes[relevant_src]  # [num_edges, hidden_dim]
                    dst_features = all_nodes[relevant_dst]  # [num_edges, hidden_dim] 
                    
                    # Transform for attention computation
                    q = self.W3[dst_type](dst_features)  # [num_edges, num_heads * head_dim]
                    k = self.W4[edge_key](src_features)  # [num_edges, num_heads * head_dim]
                    edge_k = self.W5[edge_type](relevant_edges)  # [num_edges, num_heads * head_dim]
                    
                    # Reshape for multi-head attention
                    q = q.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
                    k = (k + edge_k).view(-1, self.num_heads, self.head_dim)
                    
                    # Compute attention weights
                    attention_scores = torch.sum(q * k, dim=-1) / (self.head_dim ** 0.5)  # [num_edges, num_heads]
                    attention_weights = F.softmax(attention_scores, dim=-1)  # [num_edges, num_heads]
                    
                    # Compute values
                    v = self.W2[edge_key](src_features) + self.W5[edge_type](relevant_edges)  # [num_edges, hidden_dim]
                    v = v.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
                    
                    # Apply attention
                    attended = torch.sum(attention_weights.unsqueeze(-1) * v, dim=1)  # [num_edges, head_dim * num_heads]
                    attended = attended.view(-1, self.hidden_dim)  # [num_edges, hidden_dim]
                    
                    # Aggregate to destination nodes (handle multiple edges to same node)
                    updated_features.index_add_(0, relevant_dst, attended)
        
        # Apply layer normalization per node type
        node_start_idx = 0
        output_features = {}
        
        for node_type in self.node_types:
            node_count = node_features[node_type].size(0)
            node_end_idx = node_start_idx + node_count
            
            # Extract and normalize features for this node type
            type_features = updated_features[node_start_idx:node_end_idx]
            normalized_features = self.layer_norm[node_type](type_features)
            output_features[node_type] = normalized_features
            
            node_start_idx = node_end_idx
            
        return output_features


# Example usage:
def create_sinusoidal_edge_features(positions_i, positions_j, max_freq=10):
    """Create sinusoidal edge features from relative positions"""
    rel_pos = positions_j - positions_i  # [num_edges, 3]
    
    edge_features = []
    for freq in range(max_freq):
        edge_features.append(torch.sin(2**freq * torch.pi * rel_pos))
        edge_features.append(torch.cos(2**freq * torch.pi * rel_pos))
    
    return torch.cat(edge_features, dim=-1)  # [num_edges, 60]


# Example instantiation
if __name__ == "__main__":
    # Define node and edge types for Instant Policy
    node_types = ['scene_point', 'gripper_current', 'gripper_demo', 'gripper_action']
    edge_types = ['spatial', 'temporal', 'demo_to_current']
    
    model = HeterogeneousGraphTransformer(
        node_types=node_types,
        edge_types=edge_types,
        hidden_dim=1024,
        num_heads=16,
        head_dim=64
    )
    
    # Example input
    batch_size = 1
    node_features = {
        'scene_point': torch.randn(16, 1024),      # 16 scene points
        'gripper_current': torch.randn(6, 1024),   # 6 current gripper nodes  
        'gripper_demo': torch.randn(20, 1024),     # 20 demo gripper nodes
        'gripper_action': torch.randn(8, 1024)     # 8 action nodes
    }
    
    # Example edge indices (source, target node indices)
    edge_indices = {
        'spatial': torch.tensor([[0, 1, 2], [16, 17, 18]]),  # scene to gripper edges
        'temporal': torch.tensor([[16, 17], [17, 18]]),       # temporal gripper edges  
        'demo_to_current': torch.tensor([[22, 23], [16, 17]]) # demo to current edges
    }
    
    # Example edge features
    edge_features = {
        'spatial': torch.randn(3, 60),      # sinusoidal position encoding
        'temporal': torch.randn(2, 60),     
        'demo_to_current': torch.randn(2, 60)
    }
    
    # Node type indices
    node_types_per_node = torch.cat([
        torch.zeros(16, dtype=torch.long),   # scene points
        torch.ones(6, dtype=torch.long),     # current gripper
        torch.full((20,), 2, dtype=torch.long),  # demo gripper
        torch.full((8,), 3, dtype=torch.long)    # action gripper
    ])
    
    # Forward pass
    output = model(node_features, edge_indices, edge_features, node_types_per_node)
    
    print("Output shapes:")
    for node_type, features in output.items():
        print(f"{node_type}: {features.shape}")

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



