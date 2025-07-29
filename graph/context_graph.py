from .local_graph import *
from typing import List 

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
    

