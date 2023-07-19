import torch
import torch.nn as nn
import torch.nn.functional as F

class NonDominatedApproximator(nn.Module):
    def __init__(self, nS, nA, nO, lr=1e-4, tau=1., copy_every=100, clamp=None, device='cpu'):
        super(NonDominatedApproximator, self).__init__()
        self.nS = nS
        self.nA = nA
        self.nO = nO
        self.lr = lr
        self.tau = tau
        self.copy_every = copy_every
        self.clamp = clamp
        self.device = device
        

        self.fc1 = nn.Linear(nS, 32)
        self.fc2 = nn.Linear(1, 32)  # Updated shape for fc2 input
        self.fc3 = nn.Linear(64, 1)  # Updated output size

    def forward(self, state, point, action):
        
        state = state.view(-1, self.nS).float()  # Convert to float32
        point = point.view(-1, 1).float()  # Convert to float32
        action = action.view(-1, 1).float()  # Reshape action to [-1, 1]

        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(action))
        
        fc3_input = torch.cat((fc1, fc2), dim=1)
        output = self.fc3(fc3_input)
        return output
