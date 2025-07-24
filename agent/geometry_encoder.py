import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict

# ===========================
# UTILITY FUNCTIONS
# ===========================

def furthest_point_sampling(points, num_samples):
    """
    Efficient tensor-based FPS for 2D points
    Args:
        points: [N, 2] coordinates
        num_samples: number of points to sample
    Returns:
        sampled_indices: [num_samples] indices of selected points
    """
    N = points.shape[0]
    device = points.device
    
    if N <= num_samples:
        return torch.arange(N, device=device)
    
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    
    # Random starting point
    current_idx = torch.randint(0, N, (1,), device=device)
    sampled_indices[0] = current_idx
    
    for i in range(1, num_samples):
        # Update distances to nearest sampled point
        current_point = points[current_idx]
        new_distances = torch.norm(points - current_point, dim=1)
        distances = torch.minimum(distances, new_distances)
        
        # Select point with maximum distance to nearest sampled point
        current_idx = torch.argmax(distances)
        sampled_indices[i] = current_idx
        distances[current_idx] = 0  # Mark as selected
    
    return sampled_indices

def nearest_centroid_grouping(points, centroids, max_points_per_group=32):
    """
    Group points to their nearest centroids
    Args:
        points: [N, 2] all point coordinates
        centroids: [M, 2] centroid coordinates  
        max_points_per_group: max points per group (for padding)
    Returns:
        grouped_indices: [M, max_points_per_group] indices (-1 for padding)
        grouped_points: [M, max_points_per_group, 2] grouped point coordinates
    """
    N = points.shape[0]
    M = centroids.shape[0] 
    device = points.device
    
    if N == 0 or M == 0:
        return torch.full((M, max_points_per_group), -1, dtype=torch.long, device=device), \
               torch.zeros((M, max_points_per_group, 2), device=device)
    
    # Calculate distances from each point to each centroid
    distances = torch.cdist(points.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)  # [N, M]
    
    # Assign each point to nearest centroid
    nearest_centroid_idx = torch.argmin(distances, dim=1)  # [N]
    
    # Group points by their assigned centroid
    grouped_indices = torch.full((M, max_points_per_group), -1, dtype=torch.long, device=device)
    grouped_points = torch.zeros((M, max_points_per_group, 2), device=device)
    
    for centroid_idx in range(M):
        # Find all points assigned to this centroid
        mask = nearest_centroid_idx == centroid_idx
        assigned_points = torch.where(mask)[0]
        
        if len(assigned_points) == 0:
            # No points assigned to this centroid - use centroid itself
            grouped_points[centroid_idx, 0] = centroids[centroid_idx]
            continue
        
        # Handle variable group sizes
        if len(assigned_points) > max_points_per_group:
            # Randomly sample if too many points
            perm = torch.randperm(len(assigned_points), device=device)[:max_points_per_group]
            selected_indices = assigned_points[perm]
        else:
            # Pad by repeating points if too few
            selected_indices = assigned_points
            if len(assigned_points) < max_points_per_group:
                padding_needed = max_points_per_group - len(assigned_points)
                # Repeat the last point to fill up
                repeat_indices = assigned_points[-1].repeat(padding_needed)
                selected_indices = torch.cat([selected_indices, repeat_indices])
        
        grouped_indices[centroid_idx, :len(selected_indices)] = selected_indices
        grouped_points[centroid_idx, :len(selected_indices)] = points[selected_indices]
    
    return grouped_indices, grouped_points

def high_frequency_encoding_2d(relative_positions, num_frequencies=10):
    """
    Apply high-frequency sinusoidal encoding for 2D coordinates
    Args:
        relative_positions: [..., 2] relative coordinates
        num_frequencies: number of frequency bands
    Returns:
        encodings: [..., 4*num_frequencies] encoded positions
    """
    *batch_dims, dim = relative_positions.shape
    device = relative_positions.device
    
    encodings = []
    for freq_exp in range(num_frequencies):
        freq = (2 ** freq_exp) * torch.pi
        for d in range(dim):  # x, y for 2D
            encodings.append(torch.sin(freq * relative_positions[..., d]))
            encodings.append(torch.cos(freq * relative_positions[..., d]))
    
    return torch.stack(encodings, dim=-1)  # [..., 40]

# ===========================
# POINTNET LAYER
# ===========================

class PointNetLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 128]):
        super().__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, grouped_features):
        """
        Args:
            grouped_features: [M, K, input_dim] 
        Returns:
            aggregated_features: [M, output_dim] one feature vector per group
        """
        M, K, input_dim = grouped_features.shape
        
        # Reshape to process all points independently
        flat_features = grouped_features.view(-1, input_dim)
        
        # Apply MLP to each point independently
        point_features = self.mlp(flat_features)
        
        # Reshape back to groups
        point_features = point_features.view(M, K, -1)
        
        # Max pooling across points in each group
        aggregated_features = torch.max(point_features, dim=1)[0]
        
        return aggregated_features

# ===========================
# SET ABSTRACTION LAYER
# ===========================

class SetAbstractionLayer(nn.Module):
    def __init__(self, num_centers, max_points_per_group=32, output_dim=16):
        super().__init__()
        
        self.num_centers = num_centers
        self.max_points_per_group = max_points_per_group
        
        # PointNet processes 40-dim positional encoding (2D: 10 freq * 2 dims * 2 sin/cos = 40)
        self.pointnet = PointNetLayer(
            input_dim=40,  # High-frequency encoding for 2D
            output_dim=output_dim,
            hidden_dims=[64, 128]
        )
    
    def forward(self, points):
        """
        Args:
            points: [N, 2] point coordinates
        Returns:
            features: [num_centers, output_dim] features per centroid
            centroids: [num_centers, 2] centroid positions
        """
        if points.shape[0] == 0:
            device = points.device
            return torch.zeros(0, self.pointnet.mlp[-1].out_features, device=device), \
                   torch.zeros(0, 2, device=device)
        
        # Step 1: Sample centroids using FPS
        sampled_indices = furthest_point_sampling(points, self.num_centers)
        centroids = points[sampled_indices]  # [num_centers, 2]
        
        # Step 2: Group points to nearest centroids
        grouped_indices, grouped_points = nearest_centroid_grouping(
            points, centroids, self.max_points_per_group
        )
        
        # Step 3: Re-center points relative to their assigned centroids
        centroids_expanded = centroids.unsqueeze(1)  # [num_centers, 1, 2]
        relative_positions = grouped_points - centroids_expanded  # [num_centers, max_points_per_group, 2]
        
        # Step 4: Apply high-frequency encoding
        encoded_positions = high_frequency_encoding_2d(relative_positions)  # [num_centers, max_points_per_group, 40]
        
        # Step 5: Apply PointNet aggregation
        features = self.pointnet(encoded_positions)  # [num_centers, output_dim]
        
        return features, centroids

# ===========================
# GEOMETRY ENCODER (SIMPLE VERSION)
# ===========================

class GeometryEncoder2D(nn.Module):
    """
    Simple geometry encoder that outputs node_embd_dim features directly
    Follows Instant Policy paper: 2 Set Abstraction layers, 16 output nodes
    """
    def __init__(self, node_embd_dim=16):
        super().__init__()
        
        # Two Set Abstraction layers as described in Instant Policy paper
        self.sa_layer1 = SetAbstractionLayer(num_centers=32, output_dim=64)
        self.sa_layer2 = SetAbstractionLayer(num_centers=16, output_dim=node_embd_dim)
    
    def forward(self, point_cloud_2d):
        """
        Args:
            point_cloud_2d: [N, 2] coordinates of object pixels
        Returns:
            features: [16, node_embd_dim] geometry features 
            positions: [16, 2] centroid positions
        """
        if point_cloud_2d.shape[0] == 0:
            device = point_cloud_2d.device
            return torch.zeros(0, self.sa_layer2.pointnet.mlp[-1].out_features, device=device), \
                   torch.zeros(0, 2, device=device)
        
        # First SA layer: downsample to 32 points with 64-dim features
        features1, centroids1 = self.sa_layer1(point_cloud_2d)  # [32, 64], [32, 2]
        
        if centroids1.shape[0] == 0:
            device = point_cloud_2d.device
            return torch.zeros(0, self.sa_layer2.pointnet.mlp[-1].out_features, device=device), \
                   torch.zeros(0, 2, device=device)
        
        # Second SA layer: downsample to 16 points with node_embd_dim features
        features2, centroids2 = self.sa_layer2(centroids1)      # [16, node_embd_dim], [16, 2]
        
        return features2, centroids2

# ===========================
# OCCUPANCY NETWORK FOR TRAINING
# ===========================

class OccupancyDecoder2D(nn.Module):
    def __init__(self, feature_dim=16, pos_encoding_dim=40):
        super().__init__()
        
        input_dim = feature_dim + pos_encoding_dim  # node_embd_dim + 40
        
        # Simple MLP for occupancy prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, geometry_features, query_points):
        """
        Args:
            geometry_features: [M, node_embd_dim] features from geometry encoder
            query_points: [Q, 2] query point coordinates
        Returns:
            occupancy: [Q] occupancy predictions
        """
        if geometry_features.shape[0] == 0 or query_points.shape[0] == 0:
            return torch.zeros(query_points.shape[0], device=query_points.device)
        
        # Encode query points
        query_encoded = high_frequency_encoding_2d(query_points)  # [Q, 40]
        
        occupancy_predictions = []
        for i, query_pt in enumerate(query_points):
            # Use nearest geometry feature (simplified approach)
            feature_idx = min(i, geometry_features.shape[0] - 1)
            nearest_feature = geometry_features[feature_idx]
            
            # Combine feature with encoded query
            combined = torch.cat([nearest_feature, query_encoded[i]])
            
            # Pass through MLP
            occupancy = self.mlp(combined)
            occupancy_predictions.append(occupancy)
        
        return torch.stack(occupancy_predictions).squeeze()

class OccupancyNetwork2D(nn.Module):
    """
    Complete occupancy network for pre-training geometry encoder
    """
    def __init__(self, node_embd_dim=16):
        super().__init__()
        self.geometry_encoder = GeometryEncoder2D(node_embd_dim=node_embd_dim)
        self.occupancy_decoder = OccupancyDecoder2D(feature_dim=node_embd_dim)
    
    def forward(self, point_cloud_2d, query_points_2d):
        """
        Args:
            point_cloud_2d: [N, 2] coordinates of colored pixels
            query_points_2d: [M, 2] query coordinates
        Returns:
            occupancy: [M] occupancy predictions
        """
        # Extract geometry features
        geometry_features, centroids = self.geometry_encoder(point_cloud_2d)
        
        # Predict occupancy at query points
        occupancy = self.occupancy_decoder(geometry_features, query_points_2d)
        
        return occupancy

# ===========================
# TRAINING DATA GENERATION
# ===========================

def extract_object_pixels_from_game(pseudogame):
    """Extract object pixels from your pygame"""
    screen_pixels = pseudogame._get_screen_pixels()
    coords = np.array([[(x,y) for y in range(pseudogame.screen_height)] 
                      for x in range(pseudogame.screen_width)])
    
    # Filter out white/background pixels and agent pixels
    mask = ~(np.all(screen_pixels == [255, 255, 255], axis=2) |  # White background
             np.all(screen_pixels == [128, 0, 128], axis=2) |    # Purple agent
             np.all(screen_pixels == [0, 0, 255], axis=2))       # Blue agent
    
    if np.any(mask):
        object_coords = coords[mask]
        return torch.tensor(object_coords, dtype=torch.float32)
    else:
        return torch.zeros(0, 2, dtype=torch.float32)

def generate_query_points_2d(pseudogame, num_positive=50, num_negative=50):
    """Generate positive (on object) and negative (empty space) query points"""
    obj_rect = pseudogame.object.get_rect()
    
    positive_points = []
    negative_points = []
    
    # Generate positive points (on object boundary)
    for _ in range(num_positive):
        x = np.random.uniform(obj_rect.left, obj_rect.right)
        y = np.random.uniform(obj_rect.top, obj_rect.bottom)
        positive_points.append([x, y])
    
    # Generate negative points (empty space)
    for _ in range(num_negative):
        attempts = 0
        while attempts < 10:
            x = np.random.uniform(0, pseudogame.screen_width)
            y = np.random.uniform(0, pseudogame.screen_height)
            
            # Check if point is far enough from object
            obj_center = pseudogame.object.get_pos()['center']
            distance = np.sqrt((x - obj_center[0])**2 + (y - obj_center[1])**2)
            
            if distance > 30:  # Minimum distance from object
                negative_points.append([x, y])
                break
            attempts += 1
    
    return torch.tensor(positive_points, dtype=torch.float32), \
           torch.tensor(negative_points, dtype=torch.float32)

def generate_training_data(num_samples=1000):
    """Generate training data from your PseudoGame"""
    try:
        from tasks.twoD.game import PseudoGame
    except ImportError:
        print("Warning: Could not import PseudoGame. Please implement this function with your game.")
        return []
    
    training_data = []
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Generating sample {i}/{num_samples}")
        
        try:
            # Create random game configuration
            pseudogame = PseudoGame()
            pseudogame.draw()  # Ensure screen is rendered
            
            # Extract object pixels
            object_pixels = extract_object_pixels_from_game(pseudogame)
            
            # Generate query points
            positive_queries, negative_queries = generate_query_points_2d(pseudogame)
            
            if object_pixels.shape[0] > 0 and positive_queries.shape[0] > 0:
                training_data.append({
                    'point_cloud': object_pixels,
                    'positive_queries': positive_queries,
                    'negative_queries': negative_queries
                })
            
            # Clean up
            pseudogame._end_game()
            
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
    
    return training_data

# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_occupancy_network(num_epochs=100, lr=1e-3, node_embd_dim=16):
    """Train the occupancy network for pre-training geometry encoder"""
    print("Generating training data...")
    training_data = generate_training_data(num_samples=100)
    print(f"Generated {len(training_data)} training samples")
    
    if len(training_data) == 0:
        print("No training data generated. Please check your PseudoGame import and implementation.")
        return None
    
    # Initialize model
    model = OccupancyNetwork2D(node_embd_dim=node_embd_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Training occupancy network...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(training_data):
            point_cloud = batch['point_cloud']
            positive_queries = batch['positive_queries']
            negative_queries = batch['negative_queries']
            
            if point_cloud.shape[0] == 0:
                continue
            
            # Forward pass
            pos_pred = model(point_cloud, positive_queries)
            neg_pred = model(point_cloud, negative_queries)
            
            # Loss
            pos_target = torch.ones_like(pos_pred)
            neg_target = torch.zeros_like(neg_pred)
            
            pos_loss = F.binary_cross_entropy(pos_pred, pos_target)
            neg_loss = F.binary_cross_entropy(neg_pred, neg_target)
            loss = pos_loss + neg_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return model

# ===========================
# EXAMPLE USAGE
# ===========================

if __name__ == "__main__":
    # Example: Create and test geometry encoder
    node_embd_dim = 16  # Match your agent configuration
    geometry_encoder = GeometryEncoder2D(node_embd_dim=node_embd_dim)
    
    # Example input: some 2D coordinates
    sample_coords = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=torch.float32)
    
    # Get geometry features
    features, positions = geometry_encoder(sample_coords)
    print(f"Features shape: {features.shape}")  # Should be [16, node_embd_dim] or fewer
    print(f"Positions shape: {positions.shape}")  # Should be [16, 2] or fewer
    print(f"Feature dimension matches node_embd_dim: {features.shape[-1] == node_embd_dim}")
    
    # Example: Train occupancy network (if PseudoGame is available)
    # trained_model = train_occupancy_network(num_epochs=50, node_embd_dim=node_embd_dim)
    # if trained_model:
    #     torch.save(trained_model.geometry_encoder.state_dict(), 'geometry_encoder_2d.pth')
    
    print("Geometry encoder ready for integration with Instant Policy!")


    # Training
    train_occupancy_network()