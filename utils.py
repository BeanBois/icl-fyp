import torch 



# def recover_se2_action_noise(self, per_node_translation, per_node_rotation, per_node_gripper, agent_keypoints):
#     """
#     Recover SE(2) action noise from per-node adjustments using least squares alignment
    
#     Args:
#         per_node_translation: [N, 4, 2] - translation adjustments per keypoint
#         per_node_rotation: [N, 4, 1] - rotation effect adjustments per keypoint  
#         per_node_gripper: [N, 4, 1] - gripper state adjustments per keypoint
#         agent_keypoints: [4, 2] - original keypoint positions relative to agent center
        
#     Returns:
#         action_noise: [N, 4] - SE(2) action noise [tx, ty, theta, gripper_state]
#     """
#     N, num_nodes, _ = per_node_translation.shape
#     action_noise = torch.zeros(N, 4, device=self.device)
    
#     for i in range(N):
#         # Get the total displacement for each keypoint
#         keypoint_displacements = per_node_translation[i] + per_node_rotation[i].expand(-1, 2)  # [4, 2]
        
#         # Original keypoint positions (relative to agent center)
#         original_keypoints = agent_keypoints  # [4, 2]
        
#         # New keypoint positions after displacement
#         new_keypoints = original_keypoints + keypoint_displacements  # [4, 2]
        
#         # Solve for SE(2) transformation: new = R @ original + t
#         # This is equivalent to the SVD step in the paper but for 2D
#         se2_transform = self._solve_se2_alignment(original_keypoints, new_keypoints)
        
#         # Extract translation and rotation from SE(2) matrix
#         action_noise[i, 0] = se2_transform[0, 2]  # tx
#         action_noise[i, 1] = se2_transform[1, 2]  # ty  
#         action_noise[i, 2] = torch.atan2(se2_transform[1, 0], se2_transform[0, 0])  # theta
        
#         # Gripper state: average across all keypoints
#         action_noise[i, 3] = per_node_gripper[i].mean()
    
#     return action_noise
def _SE2_to_actions(se2,device):
    # actions from 3x3 matrix transformation => (x,y,theta)
    num_actions = se2.shape[0]
    actions = torch.zeros(num_actions, 3, device=device)
    
    # Extract translation components
    actions[:, 0] = se2[:, 0, 2]  # x_trans
    actions[:, 1] = se2[:, 1, 2]  # y_trans
    
    # Extract rotation angle and convert to degrees
    angles_rad = torch.atan2(se2[:, 1, 0], se2[:, 0, 0])
    actions[:, 2] = angles_rad
    
    return actions

def _actions_to_SE2(actions,device):
    num_actions = actions.shape[0]
    se2_actions = torch.zeros(num_actions, 3,3, device=device) # 2x2 rot matr, with 2x1 translation mat
    angles = actions[:, 2]
    x_trans = actions[:, 0]
    y_trans = actions[:, 1]

    se2_actions[:, 0, 2] = x_trans
    se2_actions[:, 1, 2] = y_trans
    se2_actions[:, 2, 2] = 1

    _sin_theta = torch.sin(angles)
    _cos_theta = torch.cos(angles)
    se2_actions[:, 0, 0] = _cos_theta
    se2_actions[:, 0, 1] = - _sin_theta
    se2_actions[:, 1, 0] = _sin_theta
    se2_actions[:, 1, 1] = _cos_theta

    return se2_actions