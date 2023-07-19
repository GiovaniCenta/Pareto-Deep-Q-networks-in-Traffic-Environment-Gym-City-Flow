
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardApproximator(nn.Module):
    def __init__(self, nS, nA, nO, lr=1e-4, tau=1., copy_every=100, clamp=None, device='cpu'):
        super(RewardApproximator, self).__init__()
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
        self.fc3 = nn.Linear(64, nO)  # Updated output size

    def forward(self, state, action):
        
        state = state.view(-1, self.nS).float()  # Convert to float32
        
        action = action.view(-1, 1).float()  # Reshape action to [-1, 1]

        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(action))
        
        fc3_input = torch.cat((fc1, fc2), dim=1)
        output = self.fc3(fc3_input)
        return output



    def should_copy(self, step):
        return self.copy_every and not step % self.copy_every

    def update_target(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

    def predict(self, model,*net_args, use_target_network=False):
        
        net = self.target_model if use_target_network else self.model
        
        preds = net(*[torch.from_numpy(a).to(self.device) for a in net_args])
        return preds

    def estimator(self, model,*net_args, use_target_network=False):
        self.model = model
        return self.predict(*net_args, use_target_network=use_target_network).detach().cpu().numpy()

    def update(self, targets, *net_args, step=None):
        self.opt.zero_grad()

        preds = self.predict(*net_args, use_target_network=False)
        l = self.loss(preds, torch.from_numpy(targets).to(self.device))
        if self.clamp is not None:
            l = torch.clamp(l, min=-self.clamp, max=self.clamp)
        l = l.mean()

        l.backward()

        if step % 100 == 0:
            total_norm = 0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.total_norm = total_norm

        self.opt.step()

        if self.should_copy(step):
            self.update_target(self.tau)

        return l