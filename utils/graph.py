import numpy as np

from enum import Enum
from typing import List
import copy
import math

# from ..tasks.twoD.game import Action, PlayerState # use when running this file
from tasks.twoD.game import Action, PlayerState # use when running from rwd


# Many fixes in the graph representation:
# 1) node features is separate from Pt. we should use Pt to get node-features.
# 2) embeddings for temporal edges dont make sense use nn.Embeddings.
# 3) change Action graph s.t it is able to account for multiple prediction steps 
# 4) think about dimension of node embeddings and edge embeddings 
    # we set node embeddings to 4 and edge embedddings to 4 for now





# change to AGENT, ATTRACTORS AND AVOIDERS 
class NodeType(Enum):
    AGENT = b'000'
    EDIBLE = b'001'
    OBSTACLE = b'010'
    GOAL = b'100'

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


def SinCosEdgeEmbedding(source, dest, D = 3):

    num_feature = source.shape[0]
    embedding = np.zeros((num_feature, 2 * D))
    diff = dest - source 
    aux_func = lambda d : np.array([np.sin(2**d  * np.pi * diff), np.cos(2**d  * np.pi * diff)]) 

    for d in range(D):
        embedding[:,d:d+2] =  aux_func(d)
    return embedding


class Edge:

    def __init__(self, source, dest, edge_type,  
                #  embedding_fucntion = SinCosEdgeEmbedding, 
                 weight = 1):
        self.source = source
        self.dest = dest
        self.weight = weight 
        # self.position_feature = embedding_fucntion(source.pos, dest.pos)
        self.type = edge_type
        self.rel_pos = dest.pos - source.pos 
    
    # def get_features(self):
    #     return [self.weight, self.position_feature]
    


# instead of hard feature sets like self.pos 
# store them in like a matrix or something node.features 
    

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
    
    # def get_features(self):
    #     return np.array([self.pos[0], self.pos[1], self.eaten]) 

class EdibleNode(Node):

    def __init__(self, pos,time):
        super().__init__(pos,time)
        self.type = NodeType.EDIBLE
        self.eaten = False 
    
    def set_eaten(self):
        self.eaten = True

    # def get_features(self):
    #     return np.array([self.pos[0], self.pos[1], self.tag, self.orientation]) 

class ObstacleNode(Node):

    def __init__(self, pos,time):
        super().__init__(pos,time)
        self.type = NodeType.OBSTACLE

    # def get_features(self):
    #     return np.array([self.pos[0], self.pos[1]]) 

class GoalNode(Node):

    def __init__(self, pos,time):
        super().__init__(pos,time)
        self.type = NodeType.GOAL

    # def get_features(self):
    #     return np.array([self.pos[0], self.pos[1]]) 



# This Local Graph represents a screenshot of the pointcloud at a particular timestep
class LocalGraph:

    def __init__(self, point_clouds, timestep, agent_pos, agent_orientation ,agent_state = None):
        self.num_nodes = 0
        self.timestep = timestep
        self.agent_nodes = None
        self.object_nodes = None
        self.nodes = None
        self.node_idx_dict = dict()
        self.agent_orientation = agent_orientation

        self._init_nodes(agent_pos, point_clouds) # initialise nodes, idx offsets and num_nodes

        # init agent state
        if agent_state is None:
            self.agent_state = PlayerState.EATING 
            if np.random.random() <= 0.5: 
                self.agent_state = PlayerState.NOT_EATING 
        else:
            if agent_state == 'not-eating':
                self.agent_state = PlayerState.NOT_EATING
            else:
                self.agent_state = PlayerState.EATING
        
        # init edges
        self.edges = None
        self._init_edges()
    
    def get_edges(self):
        return self.edges

    # needs to return node_dictionary and its index dicitonary
    def get_nodes(self):
        return self.nodes, self.node_idx_dict
    
    # aux functions
    # initialise point clouds into nodes
    # now i know how to add geometric features to the objects
    # basically what we do is kinda the same as how we link agent nodes (have a separate function for this)
    # what we will do is first add in the dict a list of (tl,tr,br,bl,center) and use it to do clustering algorithm (K means)
        # a fun thing about K means is that the funciton used to claculate k means and distance can be moddded to mobius operations!
    # then using the segmented set of nodes, we further reduce them 
        # first use corners to define boundary (what do we do as center? maybe can encode gemoetry | center, graph) s.t given center, we can predict geometry of object
        # then create a circular graph
        # then topologically sort the nodes with linear operations (|p1 - center| = distance (metric to used to topologically sort them {can even think of encodding vectors}) )
            # mobius operation can be used here too! 
        # then find nearest nodes to each edge and connect them 
        # graphs then becomes sparse
    def _init_nodes(self, agent_pos, point_clouds):    
        # init agent nodes
        AGENT_TAGS = [AgentNodeTag.EDGE1, AgentNodeTag.EDGE2, AgentNodeTag.EDGE3, AgentNodeTag.CENTER]

        agent_nodes = [AgentNode(coord, self.timestep, self.agent_orientation, tag) for (coord, tag) in zip(agent_pos,AGENT_TAGS)]


        # init scene nodes
        edible_pc = []
        obstacle_pc = []
        goal_pc = []
        

        if 'edible' in point_clouds.keys():
            edible_pc = point_clouds['edible']
        if 'obstacle' in point_clouds.keys():
            obstacle_pc = point_clouds['obstacle']
        if 'goal' in point_clouds.keys():
            goal_pc = point_clouds['goal']


        edible_nodes = [EdibleNode(info['coord'], self.timestep) for info in edible_pc]
        obstacle_nodes = [ObstacleNode(info['coord'], self.timestep) for info in obstacle_pc]
        goal_nodes = [GoalNode(info['coord'], self.timestep) for info in goal_pc]
        

        # set impt variables
        self.num_nodes = len(agent_nodes) + len(edible_nodes)  + len(obstacle_nodes) + len(goal_nodes) 
        self.agent_nodes = np.array(agent_nodes)
        self.object_nodes = np.array(edible_nodes + obstacle_nodes + goal_nodes)
        self.nodes = np.array(agent_nodes + edible_nodes + obstacle_nodes + goal_nodes) # this implicitly implies that agent nodes index will be first

        # now index nodes
        for (i,node) in enumerate(self.nodes):
            self.node_idx_dict[node] = i
    
    # according to image, it seems to be a fully connected graph
    # edge[i,j] means i connected to j
    # for agent -> edibles edge points from agent to edibles
    # for agent -> goals edge points from agent to goals
    # for agent -> obstacles edge points from obstacles to agent
    # for edible1 to edible2, edge points bothways
    def _init_edges(self):
        edges = []

        # first link all agent_nodes to object_nodes 1 direction
        for agent_node in self.agent_nodes:
            for obj_node in self.object_nodes:
                edges.append(Edge(source = agent_node, dest = obj_node, edge_type = EdgeType.OBJECT_TO_AGENT))

        # then link inter agent nodes 
        for i in range(len(self.agent_nodes)):
            for j in range(len(self.agent_nodes)):
                if i != j:
                    edges.append(
                        Edge(source = self.agent_nodes[i], dest = self.agent_nodes[j], edge_type = EdgeType.AGENT_TO_AGENT)
                      )

        # then link inter object node
        for i in range(len(self.object_nodes)):
            for j in range(len(self.object_nodes)):
                if i != j:
                    edges.append(
                        Edge(source = self.object_nodes[i], dest = self.object_nodes[j], edge_type = EdgeType.OBJECT_TO_OBJECT)
                    )

        self.edges = edges


    def draw_graph(self):
        import networkx as nx 
        import matplotlib.pyplot as plt
        G = nx.DiGraph()


        edges = [(f'{self.node_idx_dict[edge.source]}', f'{self.node_idx_dict[edge.dest]}') for edge in self.edges]
        # nodes = [f'{node_idx_dict[node]}' for node in self.nodes]
        agent_nodes = [f'{self.node_idx_dict[node]}' for node in self.nodes if type(node) == AgentNode]
        edible_nodes = [f'{self.node_idx_dict[node]}' for node in self.nodes if type(node) == EdibleNode]
        obstacle_nodes = [f'{self.node_idx_dict[node]}' for node in self.nodes if type(node) == ObstacleNode]
        goal_nodes = [f'{self.node_idx_dict[node]}' for node in self.nodes if type(node) == GoalNode]
        
        G.add_nodes_from(agent_nodes, bipartite=0)
        G.add_nodes_from(edible_nodes, bipartite=1)
        G.add_nodes_from(obstacle_nodes, bipartite=2)
        G.add_nodes_from(goal_nodes, bipartite=3)
        # G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # position nodes
        pos = {}
        for i,node in enumerate(agent_nodes):
            pos[node] = (0,i)
        for i,node in enumerate(edible_nodes):
            pos[node] = (2,i)
        for i,node in enumerate(obstacle_nodes):
            pos[node] = (4,i)
        for i,node in enumerate(goal_nodes):
            pos[node] = (6,i)

        # Draw the graph
        plt.figure(figsize=(10, 6))

        nx.draw_networkx_nodes(G, pos, nodelist=agent_nodes, node_color='lightblue', 
                            node_size=1000, label='Agents')
        nx.draw_networkx_nodes(G, pos, nodelist=edible_nodes, node_color='lightcoral', 
                            node_size=1000, label='Edibles')
        nx.draw_networkx_nodes(G, pos, nodelist=obstacle_nodes, node_color='red', 
                    node_size=1000, label='Obstacles')
        nx.draw_networkx_nodes(G, pos, nodelist=goal_nodes, node_color='yellow', 
                            node_size=1000, label='Goal')
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        plt.title("Simple Undirected Graph")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# This Demo graph is used to represent a sequence of graph
# Sequence of graph is generated through pointclouds collected from demonstrations 
class DemoGraph:
    def __init__(self, graph_frames: List[LocalGraph]):
        self.temporal_nodes = None
        # self.node_idx_dict = dict()
        self.num_nodes = 0
        self._init_nodes(graph_frames)
        self.graphs = graph_frames
        self.L = len(graph_frames)
        # self.temporal_edges = np.zeros((L,N,N)) # should we utilise a temporal edge?
        self.temporal_edges = None 
        # self.edge_idx_dict = dict()
        self._link_all_graph_frames()
    

    def get_temporal_nodes(self):
        return self.temporal_nodes #, self.node_idx_dict

    def get_temporal_edges(self):
        return self.temporal_edges #, self.edge_idx_dict
    
    def _init_nodes(self, graph_frames):
        self.temporal_nodes = []
        for graph in graph_frames:
            for agent_node in graph.agent_nodes:
                self.temporal_nodes.append(agent_node)
                self.num_nodes += 1

        # for (i,node) in enumerate(self.temporal_nodes):
            # self.node_idx_dict[node] = i


    # This function creates a demo from graph frames by linking agent nodes across time
    # link is made to represent actions, so respective agent nodes must be linked to the same node across board
    # this is why agent nodes are made 'constant' in their own coordinate system
    def _link_all_graph_frames(self):
        self.temporal_edges = set()
        for i in range(len(self.graphs) - 1):
            curr_graph = self.graphs[i]
            next_graph = self.graphs[i+1]

            # now link gripper nodes of both graphs
            self._link_graph_frames(curr_graph, next_graph)

    def _link_graph_frames(self, curr_graph, next_graph):
        curr_graph_agent_nodes = curr_graph.agent_nodes
        next_graph_agent_nodes = next_graph.agent_nodes

        for curr_node in curr_graph_agent_nodes:
            for next_node in next_graph_agent_nodes:
                if curr_node.tag == next_node.tag:
                    # next node to curr node so info can flow into the next node
                    edge = Edge(source = next_node, dest = curr_node, edge_type=EdgeType.AGENT_DEMO_AGENT)
                    
                    # self.edge_idx_dict[edge] = (self.node_idx_dict[curr_node], self.node_idx_dict[next_node])
                    self.temporal_edges.add(edge)
                    

 
# Context graph is constructed with 2 components : current grpah and a list of demographs 
class ContextGraph:

    def __init__(self, current_graph : LocalGraph, demo_graphs : List[DemoGraph]):
        self.current_graph = current_graph
        self.demo_graphs = demo_graphs
        self.temporal_nodes = []
        self.node_idx_dict = dict()
        self._init_nodes()
        

        # self.edge_idx_dict = dict()
        self.temporal_edges = set()
        self._account_for_demographs_edges()
        self._connect_demo_graphs_to_curr_graph()
     
    def get_temporal_nodes(self):
        return self.temporal_nodes, self.node_idx_dict

    def get_temporal_edges(self):
        return self.temporal_edges

    def _init_nodes(self):
        for agent_node in self.current_graph.agent_nodes:  
            self.temporal_nodes.append(agent_node)  
        for demo_graph in self.demo_graphs:
            demo_graph_nodes = demo_graph.get_temporal_nodes()
            for agent_node in demo_graph_nodes:
                self.temporal_nodes.append(agent_node)  
        for (i, node) in enumerate(self.temporal_nodes):
            self.node_idx_dict[node] = i

    # change it s.t temporal edges of demo graph and their indexing is accounted for
    def _connect_demo_graphs_to_curr_graph(self):
        for demo_graph in self.demo_graphs:
            self._link_demo_graph_to_current(demo_graph)
    
    def _link_demo_graph_to_current(self, demo_graph):
        curr_graph_agent_nodes = self.current_graph.agent_nodes
        demo_graph_agent_nodes = demo_graph.get_temporal_nodes()

        # link curr_node to demo_node
        for demo_node in demo_graph_agent_nodes:
            for curr_node in curr_graph_agent_nodes:    
                if curr_node.tag == demo_node.tag:
                    edge = Edge(source = curr_node, dest = demo_node, edge_type=EdgeType.AGENT_COND_AGENT)
                    # edge = Edge(source = demo_node, dest = curr_node, edge_type=EdgeType.AGENT_COND_AGENT)
                    self.temporal_edges.add(edge)
                    # self.edge_idx_dict[edge] = (self.node_idx_dict[demo_node], self.node_idx_dict[curr_node])

    def _account_for_demographs_edges(self):
        for demo_graph in self.demo_graphs:
            temporal_edges = demo_graph.get_temporal_edges()
            for edge in temporal_edges:
                self.temporal_edges.add(edge)
                # self.edge_idx_dict[edge] = (self.node_idx_dict[edge.source], self.node_idx_dict[edge.dest])
    


# can just take curr graph instead of Context Graph
#  we change this to take curr graph now
class ActionGraph:

    def __init__(self, curr_graph : LocalGraph, action : Action):
        self.curr_graph = curr_graph

        self.action = action
        self.moving_action = action.movement_as_vector()
        self.change_state_action = action.state_change

        # make predicted graph
        self.predicted_graph = None
        self._apply_action_to_curr_graph()

        # initialise nodes
        self.action_nodes = None 
        self.node_idx_dict = dict()
        self._init_nodes()
        self.num_nodes = len(self.action_nodes)

        self.action_edges = set() 
        self._connect_curr_graph_to_predicted_graph()

        # self._connect_demo_to_predicted_graph() # i dont think we connect demo node to future predicted action
        pass

    def get_action_nodes(self):
        return self.action_nodes, self.node_idx_dict

    def get_action_edges(self):
        return self.action_edges


    def _init_nodes(self):
        self.action_nodes = []
        for node in self.curr_graph.agent_nodes:
            self.action_nodes.append(node)
        for node in self.predicted_graph.agent_nodes:
            self.action_nodes.append(node)
        for (i, node) in enumerate(self.action_nodes):
            self.node_idx_dict[node] = i
    

    def _connect_curr_graph_to_predicted_graph(self):
        curr_graph_agent_nodes = self.curr_graph.agent_nodes
        predicted_graph_agent_nodes = self.predicted_graph.agent_nodes

        for curr_node in curr_graph_agent_nodes:
            for pred_node in predicted_graph_agent_nodes:
                if pred_node.tag == curr_node.tag:
                    # pred_node to curr_node
                    edge = Edge(source = pred_node, dest = curr_node, edge_type=EdgeType.AGENT_TIME_ACTION_AGENT)
                    # edge = Edge(source = curr_node, dest = pred_node, edge_type=EdgeType.AGENT_TIME_ACTION_AGENT)
                    self.action_edges.add(edge)
                    # self.edge_idx_dict[edge] = (self.node_idx_dict[curr_node], self.node_idx_dict[pred_node])



    def _apply_action_to_curr_graph(self):

        # first we duplicate the current local graph 
        predicted_graph = copy.deepcopy(self.curr_graph)

        # check if action changes state of agent
        # if self.change_state_action
            #

        # then we apply the actions to agent nodes of the predicted graphs 
        # we can do naive translation first, but we need to do collision checks with obstacles!
        predicted_agent_nodes = []
        for agent_node in predicted_graph.agent_nodes:
            # apply transformations and rotations
            predicted_agent_node = self._apply_action_to_agent_node(agent_node)
            predicted_agent_nodes.append(predicted_agent_node)

        # then depending on the state of the agent, we update the state of edible node with set_eaten()
        # this too is a collision check
        # check only done if agent is in eating state
        if predicted_graph.agent_state == PlayerState.EATING:
            for predicted_agent_node in predicted_agent_nodes:
                for obj_node, i in enumerate(predicted_graph.object_nodes):
                    if type(obj_node) == EdibleNode:
                        collided, collision_angle = self._check_collision(predicted_agent_nodes, obj_node)
                        if collided:
                            predicted_graph.object_nodes[i].set_eaten()

        # dont care about collisions for now
        # with updated agent_nodes, we need to see if the updated nodes collides with any obstacles. 
        # if such is the case, we need to update all the predicted nodes s.t its 'shape' does not change
        # collisions = []
        # for predicted_agent_node in predicted_agent_nodes:
        #     for obj_node in predicted_graph.object_nodes:
        #         if type(obj_node) == ObstacleNode:
        #             collided, collision_angle = self._check_collision(predicted_agent_nodes, obj_node)
        #             if collided:
        #                 collisions.append((collision_angle))
            
        # if len(collisions) > 0:
            # predicted_agent_nodes = self._update_node_pos_based_on_collisions(predicted_agent_nodes,collisions) # this fucntion is hard to implement bffr
        
        predicted_graph.agent_nodes = np.array(predicted_agent_nodes)
        self.predicted_graph = predicted_graph

        if self.change_state_action == PlayerState.EATING:
            self.predicted_graph.agent_state = PlayerState.EATING
        elif self.change_state_action == PlayerState.NOT_EATING:
            self.predicted_graph.agent_state = PlayerState.NOT_EATING


    def _apply_action_to_agent_node(self, agent_node):
        # complete this funciton 
        agent_node.pos = agent_node.pos + self.moving_action
        agent_node.orientation += self.action.rotation
        agent_node.orientation = (agent_node.orientation % 360)
        return agent_node

    # not used for now

    # def _check_collision(self, moving_node, static_node, collision_delta = 5):
    #     # moving node is the one to be update
    #     # collision delta gives us a radius that allows of uncertainty in node position wrt to object position
    #     collided = False 
    #     collision_angle = None 
    #     dxdy = static_node.pos - moving_node.pos 
    #     distance = np.linalg.norm(dxdy)
    #     if distance <= collision_delta:
    #         collided = True
    #         collision_angle = np.arctan2(dxdy[1], dxdy[0])
    #     return collided, collision_angle
    
    # # this is a naive implementation, we just raw update each collision individually s.t we assume updates dont cause further collision
    # # even if it does, it just an approximate solution and the probability of it happening is quite low
    # # our assumption is that collision should be usually 1, rarely 2 
    # def _update_node_pos_based_on_collisions(self, predicted_agent_nodes, collisions, collision_update = 1):
    #     for collision_angle in collisions:
    #         for agent_node, i in enumerate(predicted_agent_nodes):
    #             dxdy = collision_update * np.array([np.cos(collision_angle), np.sin(collision_angle)])
    #             predicted_agent_nodes[i].pos = agent_node.pos - dxdy 

    #     return predicted_agent_nodes

    
    # def _connect_demo_to_predicted_graph(self):
    #     for demo in self.curr_graph.demo_graphs:
    #         for demo_graph in demo:
    #             predicted_graph_agent_nodes = self.predicted_graph.agent_nodes
    #             demo_graph_agent_nodes = demo_graph.agent_nodes

    #             for demo_node in demo_graph_agent_nodes:
    #                 for pred_node in predicted_graph_agent_nodes:
    #                     if pred_node.tag == demo_node.tag:
    #                         self.action_edges.add(Edge(source = demo_node, dest = pred_node, edge_type=EdgeType.AGENT_DEMOACTION_AGENT))


# not used
class ObjectGraph:

    def __init__(self, agent_nodes, keypoints):

        pass

# TODO : SA changes here
def make_localgraph(obs):
    point_clouds = obs['point-clouds']
    agent_pos = obs['agent-pos']
    agent_state = obs['agent-state']
    agent_orientation = obs['agent-orientation']
    timestep = obs['time']
    graph = LocalGraph(point_clouds, timestep=timestep, agent_pos=agent_pos, agent_state=agent_state, agent_orientation=agent_orientation)
    return graph 

if __name__ == "__main__":
    # run python -m utils.graph
    from tasks.twoD.game import GameInterface

    gi = GameInterface(num_sampled_points=10, num_edibles=2, num_obstacles=2)
    obs = gi.start_game()
    point_clouds = obs['point-clouds']
    agent_pos = obs['agent-pos']
    agent_orientation = obs['agent-orientation']
    agent_state = obs['agent-state']
    timestep = obs['time']
    graph = LocalGraph(point_clouds, timestep=timestep, agent_pos=agent_pos, agent_state=agent_state, agent_orientation=agent_orientation)

    graph.draw_graph()

    action = None
    obs = gi.step()
    point_clouds = obs['point-clouds']
    agent_pos = obs['agent-pos']
    timestep = obs['time']
    agent_orientation = obs['agent-orientation']
    agent_state = obs['agent-state']
    graph = LocalGraph(point_clouds, timestep=timestep, agent_pos=agent_pos, agent_state=agent_state, agent_orientation=agent_orientation)


    graph.draw_graph()