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

        # first we duplicate the current local graph 
        predicted_graph = copy.deepcopy(self.curr_graph)

        # check if action changes state of agent

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

        if self.change_state_action == PlayerState.EATING.value:
            self.predicted_graph.agent_state = PlayerState.EATING
        elif self.change_state_action == PlayerState.NOT_EATING.value:
            self.predicted_graph.agent_state = PlayerState.NOT_EATING


    def _apply_action_to_agent_node(self, agent_node):
        
        agent_node.pos = agent_node.pos + self.moving_action
        agent_node.orientation += np.rad2deg(self.rotating_action)
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

