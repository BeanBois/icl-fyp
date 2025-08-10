
from .graph_aux import *
import numpy as np 

# change this to change task being ran 
from tasks2d import LousyPacmanPlayerState as PlayerState
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

        # not needed
        # then link inter agent nodes 
        # for i in range(len(self.agent_nodes)):
        #     for j in range(len(self.agent_nodes)):
        #         if i != j:
        #             edges.append(
        #                 Edge(source = self.agent_nodes[i], dest = self.agent_nodes[j], edge_type = EdgeType.AGENT_TO_AGENT)
        #               )

        # dont need this the occ net implicitly does this 
        # # then link inter object node
        # for i in range(len(self.object_nodes)):
        #     for j in range(len(self.object_nodes)):
        #         if i != j:
        #             edges.append(
        #                 Edge(source = self.object_nodes[i], dest = self.object_nodes[j], edge_type = EdgeType.OBJECT_TO_OBJECT)
        #             )

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

