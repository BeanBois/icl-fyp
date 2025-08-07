# can just take curr graph instead of Context Graph
#  we change this to take curr graph now
from .local_graph import *
import copy 
from tasks2d import LousyPacmanPseudoGameAction as Action 

class ActionGraph:

    def __init__(self, curr_graph : LocalGraph, action : Action):
        self.curr_graph = curr_graph

        _temp = action.as_vector(mode='rad')
        # dont really need all this can remove
        self.moving_action = _temp[:2]
        self.rotating_action = _temp[2]
        self.change_state_action = _temp[3]

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
        # Deep copy the current graph
        predicted_graph = copy.deepcopy(self.curr_graph)
        
        # DON'T move agent nodes - keep them at current position
        # predicted_graph.agent_nodes stays the same
        
        # Instead, apply INVERSE transformation to object nodes
        inverse_moving_action = -self.moving_action
        inverse_rotating_action = -self.rotating_action
        
        # Transform all object nodes by inverse action
        for i, obj_node in enumerate(predicted_graph.object_nodes):
            predicted_graph.object_nodes[i] = self._apply_inverse_action_to_object_node(
                obj_node, inverse_moving_action, inverse_rotating_action
            )
        
        # Handle state changes as before
        if self.change_state_action == PlayerState.EATING.value:
            predicted_graph.agent_state = PlayerState.EATING
        elif self.change_state_action == PlayerState.NOT_EATING.value:
            predicted_graph.agent_state = PlayerState.NOT_EATING
        
        self.predicted_graph = predicted_graph

    def _apply_inverse_action_to_object_node(self, obj_node, inverse_move, inverse_rotate):
        # Move object by inverse translation
        obj_node.pos = obj_node.pos + inverse_move
        
        # Apply inverse rotation around agent center if needed
        # (you might need to implement rotation logic here)
        
        return obj_node

    # def _apply_action_to_curr_graph(self):
    #     # first we duplicate the current local graph 
    #     predicted_graph = copy.deepcopy(self.curr_graph)

    #     # then we apply the actions to agent nodes of the predicted graphs 
    #     # we can do naive translation first, but we need to do collision checks with obstacles!
    #     predicted_agent_nodes = []
    #     for agent_node in predicted_graph.agent_nodes:
    #         # apply transformations and rotations
    #         predicted_agent_node = self._apply_action_to_agent_node(agent_node)
    #         predicted_agent_nodes.append(predicted_agent_node)

    #     predicted_graph.agent_nodes = np.array(predicted_agent_nodes)
    #     self.predicted_graph = predicted_graph

    #     if self.change_state_action == PlayerState.EATING.value:
    #         self.predicted_graph.agent_state = PlayerState.EATING
    #     elif self.change_state_action == PlayerState.NOT_EATING.value:
    #         self.predicted_graph.agent_state = PlayerState.NOT_EATING


    # def _apply_action_to_agent_node(self, agent_node):
        
    #     agent_node.pos = agent_node.pos + self.moving_action
    #     agent_node.orientation += np.rad2deg(self.rotating_action)
    #     agent_node.orientation = (agent_node.orientation % 360)
    #     return agent_node
