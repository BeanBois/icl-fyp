import torch
import torch.nn.functional as F

def SE2_to_se2(T, eps=1e-6):
    """
    Convert SE(2) matrix to se(2) tangent space vector using logarithmic map.
    
    Args:
        T: SE(2) transformation matrix of shape (..., 3, 3)
        eps: Small value threshold for numerical stability
    
    Returns:
        xi: se(2) tangent vector of shape (..., 3) as [rho_x, rho_y, theta]
    """
    # Handle batch dimensions
    batch_shape = T.shape[:-2]
    
    # Extract rotation matrix (top-left 2x2)
    R = T[..., :2, :2]
    
    # Extract translation vector (top-right 2x1)
    t = T[..., :2, 2]
    
    # Compute rotation angle using atan2
    theta = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    
    # Compute V_inverse matrix for translation
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Handle small angle case for numerical stability
    small_angle_mask = torch.abs(theta) < eps
    
    # V_inverse for general case
    V_inv = torch.zeros(*batch_shape, 2, 2, device=T.device, dtype=T.dtype)
    
    # For non-small angles
    not_small = ~small_angle_mask
    if torch.any(not_small):
        theta_nz = theta[not_small]
        sin_nz = sin_theta[not_small]
        cos_nz = cos_theta[not_small]
        
        # V_inv = (1/theta) * [[sin(theta), cos(theta)-1], [1-cos(theta), sin(theta)]]
        V_inv[not_small, 0, 0] = sin_nz / theta_nz
        V_inv[not_small, 0, 1] = (cos_nz - 1) / theta_nz
        V_inv[not_small, 1, 0] = (1 - cos_nz) / theta_nz
        V_inv[not_small, 1, 1] = sin_nz / theta_nz
    
    # For small angles, V_inv ≈ I - (1/2) * [[0, -theta], [theta, 0]]
    if torch.any(small_angle_mask):
        V_inv[small_angle_mask, 0, 0] = 1.0
        V_inv[small_angle_mask, 1, 1] = 1.0
        V_inv[small_angle_mask, 0, 1] = -0.5 * theta[small_angle_mask]
        V_inv[small_angle_mask, 1, 0] = 0.5 * theta[small_angle_mask]
    
    # Compute rho = V_inv @ t
    rho = torch.einsum('...ij,...j->...i', V_inv, t)
    
    # Stack to form se(2) vector [rho_x, rho_y, theta]
    xi = torch.stack([rho[..., 0], rho[..., 1], theta], dim=-1)
    
    return xi


def se2_to_SE2(xi):
    """
    Convert se(2) tangent vector to SE(2) matrix using exponential map.
    
    Args:
        xi: se(2) tangent vector of shape (..., 3) as [rho_x, rho_y, theta]
    
    Returns:
        T: SE(2) transformation matrix of shape (..., 3, 3)
    """
    batch_shape = xi.shape[:-1]
    
    rho = xi[..., :2]  # [rho_x, rho_y]
    theta = xi[..., 2]  # theta
    
    # Compute rotation matrix
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    R = torch.zeros(*batch_shape, 2, 2, device=xi.device, dtype=xi.dtype)
    R[..., 0, 0] = cos_theta
    R[..., 0, 1] = -sin_theta
    R[..., 1, 0] = sin_theta
    R[..., 1, 1] = cos_theta
    
    # Compute V matrix for translation
    eps = 1e-6
    small_angle_mask = torch.abs(theta) < eps
    
    V = torch.zeros(*batch_shape, 2, 2, device=xi.device, dtype=xi.dtype)
    
    # For non-small angles
    not_small = ~small_angle_mask
    if torch.any(not_small):
        theta_nz = theta[not_small]
        sin_nz = sin_theta[not_small]
        cos_nz = cos_theta[not_small]
        
        # V = (1/theta) * [[sin(theta), -(1-cos(theta))], [1-cos(theta), sin(theta)]]
        V[not_small, 0, 0] = sin_nz / theta_nz
        V[not_small, 0, 1] = -(1 - cos_nz) / theta_nz
        V[not_small, 1, 0] = (1 - cos_nz) / theta_nz
        V[not_small, 1, 1] = sin_nz / theta_nz
    
    # For small angles, V ≈ I + (1/2) * [[0, -theta], [theta, 0]]
    if torch.any(small_angle_mask):
        V[small_angle_mask, 0, 0] = 1.0
        V[small_angle_mask, 1, 1] = 1.0
        V[small_angle_mask, 0, 1] = -0.5 * theta[small_angle_mask]
        V[small_angle_mask, 1, 0] = 0.5 * theta[small_angle_mask]
    
    # Compute translation: t = V @ rho
    t = torch.einsum('...ij,...j->...i', V, rho)
    
    # Construct SE(2) matrix
    T = torch.zeros(*batch_shape, 3, 3, device=xi.device, dtype=xi.dtype)
    T[..., :2, :2] = R
    T[..., :2, 2] = t
    T[..., 2, 2] = 1.0
    
    return T


# Example usage and test
if __name__ == "__main__":
    # Create a test SE(2) matrix
    theta = torch.tensor([0.5, 1.0, 0.1])  # rotation angles
    tx = torch.tensor([2.0, 3.0, 0.5])     # x translation
    ty = torch.tensor([1.0, -1.0, 2.0])    # y translation
    
    # Manual construction of SE(2) matrices
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    T = torch.zeros(3, 3, 3)
    T[:, 0, 0] = cos_theta
    T[:, 0, 1] = -sin_theta
    T[:, 0, 2] = tx
    T[:, 1, 0] = sin_theta
    T[:, 1, 1] = cos_theta
    T[:, 1, 2] = ty
    T[:, 2, 2] = 1.0
    
    print("Original SE(2) matrices:")
    print(T)
    print(T.shape)

    
    # Convert to se(2)
    xi = SE2_to_se2(T)
    print("\nTangent space vectors se(2):")
    print(xi)
    print(xi.shape)

    
    # Convert back to SE(2)
    T_recovered = se2_to_SE2(xi)
    print("\nRecovered SE(2) matrices:")
    print(T_recovered)
    print(T_recovered.shape)
    
    # Check if they match
    print("\nMax difference:", torch.max(torch.abs(T - T_recovered)).item())
    
    # Test with single matrix
    single_T = T[0:1]  # Take first matrix
    single_xi = SE2_to_se2(single_T)
    print(f"\nSingle matrix test:")
    print(f"Original: {single_T.squeeze()}")
    print(f"Tangent:  {single_xi.squeeze()}")
    print(f"Recovered: {se2_to_SE2(single_xi).squeeze()}")