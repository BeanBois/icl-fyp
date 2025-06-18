import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.nn_components import HeterogeneousGraphTransformer

class InstantPolicyAgent:
    def __init__(self):
        # Define node and edge types for Instant Policy
        self.node_types = ['scene_point', 'gripper_current', 'gripper_demo', 'gripper_action']
        self.edge_types = ['spatial', 'temporal', 'demo_to_current', 'action_connection']
        
        # Initialize the three networks
        self.rho = RhoNN(
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_dim=1024,
            num_heads=16,
            head_dim=64
        )
        
        self.phi = PhiNN(
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_dim=1024,
            num_heads=16,
            head_dim=64
        )
        
        self.psi = PsiNN(
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_dim=1024,
            num_heads=16,
            head_dim=64
        )

    def forward(self, local_graphs, context_graphs, action_graphs):
        """
        Implements Equation 6: ε_θ(G^k) = ψ(G(σ(G_l^a), φ(G_c(σ(G_l^t), {σ(G_l^{1:L})}_{1}^N))))
        """
        # Step 1: Apply σ (rho) to all local subgraphs
        processed_current = self.rho(local_graphs['current'])
        processed_demo_graphs = [self.rho(demo_graph) for demo_graph in local_graphs['demos']]
        processed_action = self.rho(local_graphs['action'])
        
        # Step 2: Apply φ (phi) to context with processed demo graphs
        context_with_demos = self.phi(processed_current, processed_demo_graphs)
        
        # Step 3: Apply ψ (psi) to propagate information to action nodes
        denoising_predictions = self.psi(processed_action, context_with_demos)
        
        return denoising_predictions

# Rho Network: operates on local subgraphs G_l and propagates initial information 
# about the point cloud observations to the gripper nodes
class RhoNN(nn.Module):
    def __init__(self, node_types, edge_types, hidden_dim=1024, num_heads=16, head_dim=64):
        super(RhoNN, self).__init__()
        
        # Two-layer heterogeneous graph transformer
        self.layer1 = HeterogeneousGraphTransformer(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        self.layer2 = HeterogeneousGraphTransformer(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # Layer normalization for each node type
        self.layer_norms = nn.ModuleDict({
            node_type: nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim)
            ]) for node_type in node_types
        })

    def forward(self, graph_data):
        """
        Process local subgraph to propagate point cloud information to gripper nodes
        
        Args:
            graph_data: Dict containing:
                - node_features: Dict[str, Tensor] - features for each node type
                - edge_indices: Dict[str, Tensor] - edge connections
                - edge_features: Dict[str, Tensor] - edge attributes
                - node_types_per_node: Tensor - node type indices
        """
        node_features = graph_data['node_features']
        edge_indices = graph_data['edge_indices']
        edge_features = graph_data['edge_features']
        node_types_per_node = graph_data['node_types_per_node']
        
        # First layer with residual connection
        output1 = self.layer1(node_features, edge_indices, edge_features, node_types_per_node)
        
        # Apply layer norm and residual connection
        for node_type in output1.keys():
            if node_type in node_features:
                normed = self.layer_norms[node_type][0](output1[node_type])
                output1[node_type] = normed + node_features[node_type]
        
        # Second layer with residual connection
        output2 = self.layer2(output1, edge_indices, edge_features, node_types_per_node)
        
        # Apply layer norm and residual connection
        for node_type in output2.keys():
            if node_type in output1:
                normed = self.layer_norms[node_type][1](output2[node_type])
                output2[node_type] = normed + output1[node_type]
        
        return output2


# Phi Network: additionally propagates information through the demonstrated trajectories 
# and allows all the relevant information from the context to be gathered at the gripper nodes 
# of the current subgraph
class PhiNN(nn.Module):
    def __init__(self, node_types, edge_types, hidden_dim=1024, num_heads=16, head_dim=64):
        super(PhiNN, self).__init__()
        
        # Two-layer heterogeneous graph transformer for context processing
        self.context_layer1 = HeterogeneousGraphTransformer(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        self.context_layer2 = HeterogeneousGraphTransformer(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            node_type: nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim)
            ]) for node_type in node_types
        })
        
        # Network to combine multiple demonstration trajectories
        self.demo_aggregation = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types
        })

    def forward(self, current_graph, demo_graphs):
        """
        Process context by combining current observation with demonstration trajectories
        
        Args:
            current_graph: Output from RhoNN for current observation
            demo_graphs: List of outputs from RhoNN for demonstration trajectories
        """
        # Aggregate information from multiple demonstrations
        aggregated_demos = self._aggregate_demonstrations(demo_graphs)
        
        # Combine current observation with aggregated demonstrations
        combined_graph = self._combine_current_with_demos(current_graph, aggregated_demos)
        
        # Process through context layers
        output1 = self.context_layer1(
            combined_graph['node_features'],
            combined_graph['edge_indices'],
            combined_graph['edge_features'],
            combined_graph['node_types_per_node']
        )
        
        # Apply layer norm and residual
        for node_type in output1.keys():
            if node_type in combined_graph['node_features']:
                normed = self.layer_norms[node_type][0](output1[node_type])
                output1[node_type] = normed + combined_graph['node_features'][node_type]
        
        # Second layer
        output2 = self.context_layer2(
            output1,
            combined_graph['edge_indices'],
            combined_graph['edge_features'],
            combined_graph['node_types_per_node']
        )
        
        # Apply layer norm and residual
        for node_type in output2.keys():
            if node_type in output1:
                normed = self.layer_norms[node_type][1](output2[node_type])
                output2[node_type] = normed + output1[node_type]
        
        return {
            'node_features': output2,
            'edge_indices': combined_graph['edge_indices'],
            'edge_features': combined_graph['edge_features'],
            'node_types_per_node': combined_graph['node_types_per_node']
        }

    def _aggregate_demonstrations(self, demo_graphs):
        """Aggregate multiple demonstration trajectories"""
        if not demo_graphs:
            return None
        
        # Average demonstration features for each node type
        aggregated = {}
        for node_type in demo_graphs[0]['node_features'].keys():
            demo_features = [demo['node_features'][node_type] for demo in demo_graphs if node_type in demo['node_features']]
            if demo_features:
                # Take mean across demonstrations
                mean_features = torch.stack(demo_features, dim=0).mean(dim=0)
                aggregated[node_type] = self.demo_aggregation[node_type](mean_features)
            
        return aggregated

    def _combine_current_with_demos(self, current_graph, aggregated_demos):
        """Combine current observation with aggregated demonstrations"""
        # This creates the unified graph G_c(G_l^t, {G_l^{1:L}}_1^N)
        # Implementation depends on specific graph structure
        # For simplicity, concatenating node features here
        
        combined_features = {}
        combined_edge_indices = current_graph['edge_indices'].copy()
        combined_edge_features = current_graph['edge_features'].copy()
        
        # Combine node features
        for node_type in current_graph['node_features'].keys():
            current_feat = current_graph['node_features'][node_type]
            
            if aggregated_demos and node_type in aggregated_demos:
                demo_feat = aggregated_demos[node_type]
                # Concatenate current and demo features
                combined_features[node_type] = torch.cat([current_feat, demo_feat], dim=0)
            else:
                combined_features[node_type] = current_feat
        
        # Update node type indices to account for concatenated nodes
        current_node_types = current_graph['node_types_per_node']
        if aggregated_demos:
            demo_node_types = current_graph['node_types_per_node'].clone()  # Same structure as current
            combined_node_types = torch.cat([current_node_types, demo_node_types], dim=0)
        else:
            combined_node_types = current_node_types
        
        # Add demo-to-current edges (grey edges in the paper)
        # This connects demonstration gripper nodes to current gripper nodes
        if aggregated_demos:
            self._add_demo_to_current_edges(combined_edge_indices, combined_edge_features, current_graph, aggregated_demos)
        
        return {
            'node_features': combined_features,
            'edge_indices': combined_edge_indices,
            'edge_features': combined_edge_features,
            'node_types_per_node': combined_node_types
        }

    def _add_demo_to_current_edges(self, edge_indices, edge_features, current_graph, aggregated_demos):
        """Add edges connecting demonstration nodes to current observation nodes"""
        # Implementation for adding demo-to-current connections
        # This creates the grey edges shown in Figure 2 of the paper
        pass


# Psi Network: propagates information to nodes in the graph representing the actions
class PsiNN(nn.Module):
    def __init__(self, node_types, edge_types, hidden_dim=1024, num_heads=16, head_dim=64):
        super(PsiNN, self).__init__()
        
        # Two-layer heterogeneous graph transformer for action prediction
        self.action_layer1 = HeterogeneousGraphTransformer(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        self.action_layer2 = HeterogeneousGraphTransformer(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            node_type: nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim)
            ]) for node_type in node_types
        })
        
        # Final MLP for denoising direction prediction
        # As mentioned in the paper: "features of nodes representing robot actions are processed 
        # with a 2-layer MLP equipped with GeLU to produce per-node denoising directions"
        self.denoising_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7)  # 7D output: [∇p̂_t, ∇p̂_r] for SE(3) actions
        )

    def forward(self, action_graph, context_output):
        """
        Propagate information from context to action nodes and predict denoising directions
        
        Args:
            action_graph: Graph containing action nodes (from RhoNN)
            context_output: Processed context from PhiNN
        """
        # Combine action graph with context information
        combined_graph = self._combine_action_with_context(action_graph, context_output)
        
        # Process through action layers
        output1 = self.action_layer1(
            combined_graph['node_features'],
            combined_graph['edge_indices'], 
            combined_graph['edge_features'],
            combined_graph['node_types_per_node']
        )
        
        # Apply layer norm and residual
        for node_type in output1.keys():
            if node_type in combined_graph['node_features']:
                normed = self.layer_norms[node_type][0](output1[node_type])
                output1[node_type] = normed + combined_graph['node_features'][node_type]
        
        # Second layer
        output2 = self.action_layer2(
            output1,
            combined_graph['edge_indices'],
            combined_graph['edge_features'], 
            combined_graph['node_types_per_node']
        )
        
        # Apply layer norm and residual
        for node_type in output2.keys():
            if node_type in output1:
                normed = self.layer_norms[node_type][1](output2[node_type])
                output2[node_type] = normed + output1[node_type]
        
        # Extract action node features and predict denoising directions
        action_features = output2['gripper_action']  # Extract only action node features
        denoising_directions = self.denoising_mlp(action_features)
        
        return denoising_directions

    def _combine_action_with_context(self, action_graph, context_output):
        """Combine action nodes with context information"""
        # Create unified graph with action nodes connected to context
        # This implements the final graph G(G_l^a(a), G_c(...))
        
        combined_features = {}
        combined_edge_indices = {}
        combined_edge_features = {}
        
        # Add context features
        for node_type, features in context_output['node_features'].items():
            combined_features[node_type] = features
            
        # Add action features
        for node_type, features in action_graph['node_features'].items():
            if node_type in combined_features:
                combined_features[node_type] = torch.cat([combined_features[node_type], features], dim=0)
            else:
                combined_features[node_type] = features
        
        # Combine edge information
        combined_edge_indices.update(context_output['edge_indices'])
        combined_edge_indices.update(action_graph['edge_indices'])
        combined_edge_features.update(context_output['edge_features'])
        combined_edge_features.update(action_graph['edge_features'])
        
        # Add action-to-context connections (the paper mentions edges that "propagate information 
        # from the current observation (and indirectly the context) to nodes representing actions")
        self._add_action_context_edges(combined_edge_indices, combined_edge_features, context_output, action_graph)
        
        # Combine node type indices
        context_node_count = sum(feat.size(0) for feat in context_output['node_features'].values())
        action_node_types = action_graph['node_types_per_node'] + context_node_count
        combined_node_types = torch.cat([context_output['node_types_per_node'], action_node_types], dim=0)
        
        return {
            'node_features': combined_features,
            'edge_indices': combined_edge_indices,
            'edge_features': combined_edge_features,
            'node_types_per_node': combined_node_types
        }

    def _add_action_context_edges(self, edge_indices, edge_features, context_output, action_graph):
        """Add edges connecting context to action nodes"""
        # Implementation for connecting current gripper nodes to action nodes
        # These are the edges that enable information flow from context to predicted actions
        pass