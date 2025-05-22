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

# 
class RhoNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RhoNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)


class PhiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PhiNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

class PsiNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PsiNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

