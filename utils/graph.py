import numpy as np

from enum import Enum
from typing import List
import copy
import math
from ..tasks.twoD.game import Action

class AgentState(Enum):
    EATING = 0
    NOT_EATING = 1


class NodeType(Enum):
    AGENT = 1
    EDIBLE = 2
    OBSTACLE = 3
    GOAL = 4

# 1 hot encoding
class EdgeType(Enum):
    # For Local Graphs
    AGENT_TO_AGENT = b'0000001'
    AGENT_TO_OBJECT = b'0000010'
    OBJECT_TO_OBJECT = b'0000100'

    # this type of edge links demo grippper nodes to current gripper node, used in Context Graph
    AGENT_COND_AGENT = b'0001000'
    # this type of edge links demo grippper nodes across timesteps in demo, used in Demo Graph
    AGENT_DEMO_AGENT = b'0010000'
    # this type of edge links current gripper node to predicted graphs (from current graph, with diffusion model)
    AGENT_TIME_ACTION_AGENT = b'0100000'
    # this type of edge links demo gripper node to predicted graphs (from current graph, with diffusion model)
    AGENT_DEMOACTION_AGENT = b'1000000'

class Edge:

    def __init__(self,source, dest, edge_type,  distance_metric = lambda x,y : np.abs(np.linalg.norm(x-y)), weight = 1):
        self.source = source
        self.dest = dest
        self.weight = weight 
        self.distance_feature = distance_metric(source.pos, dest.pos)
        self.type = edge_type
        self.rel_pos = dest.pos - source.pos 
        # a vector for now, in ip paper a sine/cosine emb is used 
        # sin(2^0 * pi * (p_j-p_i), cos(2^0 * pi * (p_j - p_i) ... sin(2^(D-1) ... , cos(2^(D-1)...)
        

    
    def get_features(self):
        return {
            'weight' : self.weight,
            'distance' : self.distance_feature,
            'rel-pos' : self.rel_pos
        }
    

    
class Node:

    def __init__(self,pos,t):
        self.pos = pos
        self.timestep = t

    def get_pos(self):
        return self.pos
    
    def set_pos(self,pos):
        self.pos = pos

class AgentNodeTag(Enum):
    EDGE1 = b'0001'
    EDGE2 = b'0010'
    EDGE3 = b'0100'
    CENTER = b'1000'
    

class AgentNode(Node):
    def __init__(self,pos,t, orientation, tag):
        super().__init__(pos,t)
        self.type = NodeType.AGENT
        self.tag = tag
        self.orientation = orientation

class EdibleNode(Node):

    def __init__(self, pos,t):
        super().__init__(pos,t)
        self.type = NodeType.EDIBLE
        self.eaten = False 
    
    def set_eaten(self):
        self.eaten = True 

class ObstacleNode(Node):

    def __init__(self, pos,t):
        super().__init__(pos,t)
        self.type = NodeType.OBSTACLE

class GoalNode(Node):

    def __init__(self, pos,t):
        super().__init__(pos,t)
        self.type = NodeType.GOAL




# This Local Graph represents a screenshot of the pointcloud at a particular timestep
class LocalGraph:

    def __init__(self, point_clouds, timestep, agent_pos, agent_orientation ,agent_state = None):
        self.num_nodes = 0
        # maybe can remove the odx offset since we have a node_idx_dict
        self.agent_idx_offset = 0
        self.edible_idx_offset = 0
        self.obstacle_idx_offset = 0
        self.goal_idx_offset = 0
        self.timestep = timestep

        self.agent_nodes = None
        self.object_nodes = None
        self.nodes = None
        self.node_idx_dict = dict()
        self.agent_orientation = agent_orientation
        self._init_nodes(agent_pos, point_clouds) # initialise nodes, idx offsets and num_nodes

        # init agent state
        if agent_state is None:
            self.agent_state = AgentState.EATING 
            if np.random.random() <= 0.5: 
                self.agent_state = AgentState.NOT_EATING 
        else:
            if agent_state == 'not-eating':
                self.agent_state = AgentState.NOT_EATING
            else:
                self.agent_state = AgentState.EATING
        
        # init edges
        self.edges = None
        self._init_edges()
    
    # need to convert edge objects to matrix s.t it can be used in transformers
    # this should be a N x N matrix, where N is the number of nodes
    def get_edges(self):
        # edge have attributes weight, distance-metric, edgetype
        edges = np.zeros((self.num_nodes, self.num_nodes, 3))

        for edge in self.edges:
            source_node_idx = self.node_idx_dict[edge.source]
            dest_node_idx = self.node_idx_dict[edge.dest]
            weight = edge.weight
            distance_metric = edge.distance_feature
            edge_type = edge.type
            edges[source_node_idx, dest_node_idx, :] = np.array([weight,distance_metric, edge_type])

        return edges

    # need to convert nodes into matrix s.t it can be used in transformers
    def get_nodes(self):
        nodes = np.zeros(self.num_gn + self.num_pcn)
        nodes[:self.num_gn] = self.agent_pos
        nodes[self.num_gn:] = np.array([
            pc.pos for pc in self.point_clouds
        ])
        return nodes     
    
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
        # TODO update this 
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
        
        # admin stuff to keep data integrity
        self.edible_idx_offset += len(agent_nodes) # update edible offset
        self.obstacle_idx_offset += self.edible_idx_offset + len(edible_nodes) # update obstacle offset
        self.goal_idx_offset += self.obstacle_idx_offset + len(obstacle_nodes) # update obstacle offset
        self.num_nodes = self.goal_idx_offset + len(goal_nodes) 

        # set impt variables
        self.agent_nodes = np.array(agent_nodes)
        self.object_nodes = np.array(edible_nodes + obstacle_nodes + goal_nodes)
        self.nodes = np.array(agent_nodes + edible_nodes + obstacle_nodes + goal_nodes)

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
                edges.append(Edge(source = agent_node, dest = obj_node, edge_type = EdgeType.AGENT_TO_OBJECT))

        # then link inter agent nodes (i dont thinke we interlink agent nodes)
        # for i in range(len(self.agent_nodes)):
        #     for j in range(len(self.agent_nodes)):
        #         if i != j:
        #             edges.append(
        #                 Edge(source = self.agent_nodes[i], dest = self.agent_nodes[j], edge_type = EdgeType.AGENT_TO_AGENT)
        #               )

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
        self.temporal_edges = None
        self.graphs = graph_frames
        self.L = len(graph_frames)
        self.N = len(graph_frames[0])
        # self.temporal_edges = np.zeros((L,N,N)) # should we utilise a temporal edge?
        self.temporal_edges = None 
        self._link_all_graph_frames()
        

    # Should return a L x N x node-feature-size matrix
    def get_nodes(self):
        pass

    # this should return a L x N x N matrix, where L is the lenght of sequence, N is the number of agent nodes
    def get_temporal_edges(self):
        pass


    # This function creates a demo from graph frames by linking agent nodes across time
    # link is made to represent actions, so respective agent nodes must be linked to the same node across board
    # this is why agent nodes are made 'constant' in their own coordinate system
    def _link_all_graph_frames(self):
        self.temporal_edges = []
        for i in range(len(self.graphs) - 1):
            curr_graph = self.graphs[i]
            next_graph = self.graphs[i+1]

            # now link gripper nodes of both graphs
            self._link_graph_frame(curr_graph, next_graph)

    def _link_graph_frames(self, curr_graph, next_graph):
        curr_graph_agent_nodes = curr_graph.agent_nodes
        next_graph_agent_nodes = next_graph.agent_nodes

        for curr_node in curr_graph_agent_nodes:
            for next_node in next_graph_agent_nodes:
                if curr_node.tag == next_node.tag:
                    self.temporal_edges.append(Edge(source = curr_node, dest = next_node, edge_type=EdgeType.AGENT_DEMO_AGENT))


# Context graph is constructed with 2 components : current grpah and a list of demographs 
class ContextGraph:

    def __init__(self, current_graph : LocalGraph, demo_graphs : List[DemoGraph]):
        self.current_graph = current_graph
        self.demo_graphs = demo_graphs
        
        self.demo_to_current_edges = None
        self._connect_demo_graphs_to_curr_graph()

    # this should return a N x N x L x DEMO_SIZE matrix
    def get_demo_edges(self):
        pass

    def _connect_demo_graphs_to_curr_graph(self):
        self.demo_to_current_edges = []
        
        for demos in range(len(self.demo_graphs)):
            for demo_graph in demos:
                self._link_demo_graph_to_current(demo_graph)

    def _link_graph_frames(self, demo_graph):
        curr_graph_agent_nodes = self.current_graph.agent_nodes
        demo_graph_agent_nodes = demo_graph.agent_nodes

        for demo_node in demo_graph_agent_nodes:
            for curr_node in curr_graph_agent_nodes:    
                if curr_node.tag == demo_node.tag:
                    self.temporal_edges.append(Edge(source = demo_node, dest = curr_node, edge_type=EdgeType.AGENT_COND_AGENT))


# TODO : refactor action here 
class ActionGraph:

    def __init__(self, context_graph : ContextGraph, action : Action):
        self.context_graph = context_graph

        self.moving_action = action.movement_as_matrix()
        self.change_state_action = action.state_change

        self.action_edges = [] 
        self.predicted_graph = None
        self._apply_action_to_curr_graph()
        self._connect_curr_graph_to_predicted_graph()
        # self._connect_demo_to_predicted_graph() i dont think we connect demo node to future predicted action
        pass

    def get_action_edges(self):
        # if length of predicion is T, num agent node is N, 
        # matrix should be T X N X N
        pass

    # def _connect_demo_to_predicted_graph(self):
    #     for demo in self.context_graph.demo_graphs:
    #         for demo_graph in demo:
    #             predicted_graph_agent_nodes = self.predicted_graph.agent_nodes
    #             demo_graph_agent_nodes = demo_graph.agent_nodes

    #             for demo_node in demo_graph_agent_nodes:
    #                 for pred_node in predicted_graph_agent_nodes:
    #                     if pred_node.tag == demo_node.tag:
    #                         self.action_edges.append(Edge(source = demo_node, dest = pred_node, edge_type=EdgeType.AGENT_DEMOACTION_AGENT))


    def _connect_curr_graph_to_predicted_graph(self):
        curr_graph_agent_nodes = self.context_graph.current_graph.agent_nodes
        predicted_graph_agent_nodes = self.predicted_graph.agent_nodes

        for curr_node in curr_graph_agent_nodes:
            for pred_node in predicted_graph_agent_nodes:
                if pred_node.tag == curr_node.tag:
                    self.action_edges.append(Edge(source = curr_node, dest = pred_node, edge_type=EdgeType.AGENT_TIME_ACTION_AGENT))

    def _apply_action_to_curr_graph(self):

        # first we duplicate the current local graph 
        predicted_graph = copy.deepcopy(self.context_graph.curr_graph)

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
        if predicted_graph.agent_state == AgentState.EATING:
            for predicted_agent_node in predicted_agent_nodes:
                for obj_node, i in enumerate(predicted_graph.object_nodes):
                    if type(obj_node) == EdibleNode:
                        collided, collision_angle = self._check_collision(predicted_agent_nodes, obj_node)
                        if collided:
                            predicted_graph.object_nodes[i].set_eaten()

        # with updated agent_nodes, we need to see if the updated nodes collides with any obstacles. 
        # if such is the case, we need to update all the predicted nodes s.t its 'shape' does not change
        collisions = []
        for predicted_agent_node in predicted_agent_nodes:
            for obj_node in predicted_graph.object_nodes:
                if type(obj_node) == ObstacleNode:
                    collided, collision_angle = self._check_collision(predicted_agent_nodes, obj_node)
                    if collided:
                        collisions.append((collision_angle))
            
        if len(collisions) > 0:
            predicted_agent_nodes = self._update_node_pos_based_on_collisions(predicted_agent_nodes,collisions) # this fucntion is hard to implement bffr
        
        predicted_graph.agent_nodes = np.array(predicted_agent_nodes)
        self.predicted_graph = predicted_graph

    # this function is most definitely wrong HAHAHAHA
    # fix this
    def _apply_action_to_agent_node(self, agent_node):
        # complete this funciton 
        # action consist of a rotation and scalar translation
        # we assume self.moving_action to be A = R @ T (T is forward direction [1,0], R is 2x2 matrix)
        agent_node.pos = agent_node.pos + self.moving_action

        return agent_node

    def _check_collision(self, moving_node, static_node, collision_delta = 5):
        # moving node is the one to be update
        # collision delta gives us a radius that allows of uncertainty in node position wrt to object position
        collided = False 
        collision_angle = None 
        dxdy = static_node.pos - moving_node.pos 
        distance = np.linalg.norm(dxdy)
        if distance <= collision_delta:
            collided = True
            collision_angle = np.arctan2(dxdy[1], dxdy[0])
        return collided, collision_angle
    
    # this is a naive implementation, we just raw update each collision individually s.t we assume updates dont cause further collision
    # even if it does, it just an approximate solution and the probability of it happening is quite low
    # our assumption is that collision should be usually 1, rarely 2 
    def _update_node_pos_based_on_collisions(self, predicted_agent_nodes, collisions, collision_update = 1):
        for collision_angle in collisions:
            for agent_node, i in enumerate(predicted_agent_nodes):
                dxdy = collision_update * np.array([np.cos(collision_angle), np.sin(collision_angle)])
                predicted_agent_nodes[i].pos = agent_node.pos - dxdy 

        return predicted_agent_nodes


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