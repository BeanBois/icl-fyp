import numpy as np
from types import List, Tuple
from enum import Enum

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