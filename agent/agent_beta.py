
# trying out another way 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class HeterogeneousGraphTransformer(nn.Module):
    def __init__(self, 
                 node_types: List[str],
                 edge_types: List[str], 
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
            src_types = node_types_per_node[src_nodes]
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