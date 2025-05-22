import numpy as np
from types import List, Tuple
from enum import Enum


class GripperState(Enum):
    OPENED = 0
    CLOSED = 1


def make_nodes(point_clouds : List[Tuple[np.float32]], robot_pos : List[Tuple[np.float32]]):

    point_clouds = [PointCloudNode((pc[0],pc[1],pc[2]), (pc[3],pc[4],pc[5])) for pc in point_clouds]
    robot_pos = [RobotNode((rp[0],rp[1],rp[2])) for rp in robot_pos]
    return point_clouds, robot_pos


def group_nodes_to_object(point_clouds:List[PointCloudNode], img):

    # point of this function is to group nodes together s.t pc from the same object is group together
    # this is so that different edges can be used to connect the edges
    # object Edge + inter-object Edges
    # this is done by using the image to align the point clouds

    # assume that we have a function that segments the img s.t dimensions of the object can be retrieved. Eg of dim [tl,tr,bl,br]
    # dim is coordinated wrt to the world frame, like the coords. of the point_clouds

    pass


class NodeType(Enum):
    POINTCLOUD = 1
    ROBOT = 2


class Node:

    def __init__(self,pos : Tuple[np.float32]):
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        self.pos_z = pos[2]

    def get_pos(self):
        return (self.pos_x,self.pos_y, self.pos_z)
    
    def set_pos(self,pos : Tuple[np.float32]):
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        self.pos_z = pos[2]

class PointCloudNode(Node):

    def __init__(self, pos : Tuple[np.float32], rgb : Tuple[int], ):
        super().__init__(pos)
        self.red_val = rgb[0]
        self.blue_val = rgb[1]
        self.green_val = rgb[2]
        self.type = NodeType.POINTCLOUD

class RobotNode(Node):

    def __init__(self,pos : Tuple[np.float32]):
        super().__init__(pos)
        self.type = NodeType.ROBOT




# how do we construct local graphs?
class LocalGraph:

    def __init__(self, point_clouds, gripper_points, gripper_state = None):
        self.point_clouds = self._init_pointclouds_nodes(point_clouds)
        self.end_effector_pose = self._init_robot_nodes(gripper_points)
        if gripper_state is None:
            self.gripper_state = GripperState.OPENED 
            if np.random.random() <= 0.5: 
                self.gripper_state = GripperState.CLOSED 
        else:
            self.gripper_state = gripper_state
        self.edges = self._init_edges()
    
    # edges are NxN
    def get_edges(self):
        return self.edges

    # nodes are 3xN
    def get_nodes(self):
        nodes = np.zeros(self.num_gn + self.num_pcn)
        nodes[:self.num_gn] = self.end_effector_pose
        nodes[self.num_gn:] = np.array([
            pc.pos for pc in self.point_clouds
        ])
        return nodes     
    
    # aux functions
    def _init_pointclouds_nodes(self, point_clouds):
        point_clouds_nodes = []
        for pc in point_clouds:
            pos = pc[:3]
            rgb = pc[3:]
            point_clouds_nodes.append(
                PointCloudNode(
                    pos,
                    rgb
                )
            )
        self.num_pcn = len(point_clouds_nodes)
        return np.array(point_clouds_nodes)
    
    def _init_robot_nodes(self,gripper_points):
        gripper_nodes = []
        for gp in gripper_points:
            gripper_nodes.append(
                RobotNode(
                    gp
                )
            )
        self.num_gn = len(gripper_nodes)
        return np.array(gripper_nodes) 
    
    # according to image, it seems to be a fully connected graph
    def _init_edges(self):
        return np.random.random((self.num_gn+self.num_pcn, self.num_gn+self.num_pcn))





             
class ActionGraph:

    # actions graphs are constructed from a given local_graph, where as if the actions were executed and the gripper moved: G_a_l(P, T_WE x T_ea, a_g)
    def __init__(self, local_graph,actions):
        pass

class ContextGraph:

    def __init__(self):
        self.graphs = list() # sequence of local graphs
        self.action_graphs = list() # seq of actions inbetween graphs
        self._interleave()

    def _interleave(self):
        # interleave is done by linking gripper nodes across time to represent relative movements
        # connecting all demos gripper nodes to the current ones to propagate relevant info
        pass

