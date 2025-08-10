from enum import Enum 


class NodeType(Enum):
    AGENT = b'000'
    ATTRACTOR = b'001'
    AVOIDER = b'010'

# 1 hot encoding
class EdgeType(Enum):
    # For Local Graphs
    AGENT_TO_AGENT =            b'000001' # gripper rel gripper
    OBJECT_TO_AGENT =           b'000010' # this should be scene rel gripper!
    OBJECT_TO_OBJECT =          b'000100' # scene rel scene
    AGENT_COND_AGENT =          b'001000' # gripper cond gripper this type of edge links demo grippper nodes to current gripper node, used in Context Graph
    AGENT_DEMO_AGENT =          b'010000' # gripper demo gripper this type of edge links demo grippper nodes across timesteps in demo, used in Demo Graph
    AGENT_TIME_ACTION_AGENT =   b'100000' # gripper timeaction grippEr this type of edge links current gripper node to predicted graphs (from current graph, with diffusion model)

    # this type of edge links demo gripper node to predicted graphs (from current graph, with diffusion model)
    # OBJECT_RELDEMO_AGENT =    b'000000100000' # unused for now
    # OBJECT_RELDEMO_OBJECT =   b'000001000000' # unused for now
    # AGENT_DEMOACTION_AGENT =  b'000100000000' # unused for now
    # OBJECT_RELDEMO_AGENT =    b'001000000000' # unused for now
    # OBJECT_RELDEMO_OBJECT =   b'010000000000' # unused for now
    # AGENT_RELCOND_AGENT =     b'100000000000' # unused for now


class Edge:

    def __init__(self, source, dest, edge_type, weight = 1):
        self.source = source
        self.dest = dest
        self.weight = weight 
        self.type = edge_type
        self.rel_pos = dest.pos - source.pos 
    
    

class Node:
    def __init__(self,pos,time):
        self.pos = pos
        self.timestep = time
    
    def get_pos(self):
        return self.pos

    def get_time(self):
        return self.timestep


class AgentNodeTag(Enum):
    EDGE1 = b'0001'
    EDGE2 = b'0010'
    EDGE3 = b'0100'
    CENTER = b'1000'
    
class AgentNode(Node):
    def __init__(self, pos, time, orientation, tag):
        super().__init__(pos,time)
        self.type = NodeType.AGENT
        self.tag = tag
        self.orientation = orientation
    
class AttractorNode(Node):
    def __init__(self, pos, time):
        super().__init__(pos,time)
        self.type = NodeType.ATTRACTOR
        self.passed = False 

    def set_passed(self):
        self.passed = True 

class EdibleNode(AttractorNode):

    def __init__(self, pos,time):
        super().__init__(pos,time)

class GoalNode(AttractorNode):

    def __init__(self, pos,time):
        super().__init__(pos,time)


class ObstacleNode(Node):

    def __init__(self, pos,time):
        super().__init__(pos,time)
        self.type = NodeType.AVOIDER






