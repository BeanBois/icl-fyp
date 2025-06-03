import numpy as np

from enum import Enum


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
    AGENT_TO_AGENT = b'001'
    AGENT_TO_OBJECT = b'010'
    OBJECT_TO_OBJECT = b'100'

class Edge:

    def __init__(self,source, dest, edge_type,  distance_metric = lambda x,y : np.abs(np.linalg.norm(x-y)), weight = 1):
        self.source = source
        self.dest = dest
        self.weight = weight 
        self.distance_feature = distance_metric(source.pos, dest.pos)
        self.type = edge_type

    
    def get_features(self):
        return {
            'weight' : self.weight,
            'distance' : self.distance_feature
        }
    

    
class Node:

    def __init__(self,pos):
        self.pos = pos

    def get_pos(self):
        return self.pos
    
    def set_pos(self,pos):
        self.pos = pos

class AgentNode(Node):

    def __init__(self,pos):
        super().__init__(pos)
        self.type = NodeType.AGENT

class EdibleNode(Node):

    def __init__(self, pos):
        super().__init__(pos)
        self.type = NodeType.EDIBLE

class ObstacleNode(Node):

    def __init__(self, pos):
        super().__init__(pos)
        self.type = NodeType.OBSTACLE

class GoalNode(Node):

    def __init__(self, pos):
        super().__init__(pos)
        self.type = NodeType.GOAL



# how do we construct local graphs?
class LocalGraph:

    def __init__(self, point_clouds, agent_state = None):
        self.num_nodes = 0
        self.agent_idx_offset = 0
        self.edible_idx_offset = 0
        self.obstacle_idx_offset = 0
        self.goal_idx_offset = 0

        self.agent_nodes = None
        self.object_nodes = None
        self.nodes = None
        self._init_pointclouds_nodes(point_clouds) # initialise nodes, idx offsets and num_nodes

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
    
    # edges are NxN
    def get_edges(self):
        return self.edges

    # nodes are 3xN
    def get_nodes(self):
        nodes = np.zeros(self.num_gn + self.num_pcn)
        nodes[:self.num_gn] = self.agent_pos
        nodes[self.num_gn:] = np.array([
            pc.pos for pc in self.point_clouds
        ])
        return nodes     
    
    # aux functions
    def _init_pointclouds_nodes(self, point_clouds):
        agent_pc = []
        edible_pc = []
        obstacle_pc = []
        goal_pc = []
        
        if 'agent' in point_clouds.keys():
            agent_pc = point_clouds['agent']
        if 'edible' in point_clouds.keys():
            edible_pc = point_clouds['edible']
        if 'obstacle' in point_clouds.keys():
            obstacle_pc = point_clouds['obstacle']
        if 'goal' in point_clouds.keys():
            goal_pc = point_clouds['goal']


        agent_nodes = [AgentNode(info['coord']) for info in agent_pc]
        edible_nodes = [EdibleNode(info['coord']) for info in edible_pc]
        obstacle_nodes = [ObstacleNode(info['coord']) for info in obstacle_pc]
        goal_nodes = [GoalNode(info['coord']) for info in goal_pc]
        
        self.edible_idx_offset += len(agent_nodes) # update edible offset
        self.obstacle_idx_offset += self.edible_idx_offset + len(edible_nodes) # update obstacle offset
        self.goal_idx_offset += self.obstacle_idx_offset + len(obstacle_nodes) # update obstacle offset
        self.num_nodes = self.goal_idx_offset + len(goal_nodes) 
        self.agent_nodes = np.array(agent_nodes)
        self.object_nodes = np.array(edible_nodes + obstacle_nodes + goal_nodes)
        self.nodes = np.array(agent_nodes + edible_nodes + obstacle_nodes + goal_nodes)
    
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
        G = nx.Graph()

        node_idx_dict = dict()

        for (i,node) in enumerate(self.nodes):
            node_idx_dict[node] = i

        edges = [(f'{node_idx_dict[edge.source]}', f'{node_idx_dict[edge.dest]}') for edge in self.edges]
        nodes = [f'{node_idx_dict[node]}' for node in self.nodes]

        G.add_nodes_from(nodes)

        G.add_edges_from(edges)

        # Draw the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)  # Position nodes using spring layout
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=16, font_weight='bold')
        plt.title("Simple Undirected Graph")
        plt.show()

             
class ActionGraph:

    # actions graphs are constructed from a given local_graph, where as if the actions were executed and the agent moved: G_a_l(P, T_WE x T_ea, a_g)
    def __init__(self, local_graph,actions):
        pass

class ContextGraph:

    def __init__(self):
        self.graphs = list() # sequence of local graphs
        self.action_graphs = list() # seq of actions inbetween graphs
        self._interleave()

    def _interleave(self):
        # interleave is done by linking agent nodes across time to represent relative movements
        # connecting all demos agent nodes to the current ones to propagate relevant info
        pass



if __name__ == "__main__":
    # run python -m utils.graph
    from tasks.twoD.game import GameInterface

    gi = GameInterface()
    obs = gi.start_game()
    point_clouds = obs['point-clouds']
    agent_state = obs['agent-state']
    graph = LocalGraph(point_clouds, agent_state)

    graph.draw_graph()

    action = None
    obs = gi.step()
    point_clouds = obs['point-clouds']
    agent_state = obs['agent-state']
    graph = LocalGraph(point_clouds, agent_state)

    graph.draw_graph()