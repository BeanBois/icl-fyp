import torch
import torch.nn as nn

import numpy as np



class InstantPolicyAgent:

    def __init__(self):
        pass

    def _train(self):
        pass

    def _eval(self):
        pass

# operates on local subgraphs G_l and propagates initial information about the point cloud observations to the gripper nodes
class RhoNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RhoNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)


# additionally propagates information through the demonstrated trajectories and allows all the relvant information from the context to be gathered at the gripper nodes of the current subgraph 
class PhiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PhiNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

# propagates information to nodes in the graph representing the actions
class PsiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PsiNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

