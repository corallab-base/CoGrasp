import torch
from torch import nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule

device = 'cuda'

class PruningNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointnetSAModule(npoint=128,radius=0.025, nsample=64, mlp=[4, 64, 64, 128]))
        self.SA_modules.append(PointnetSAModule(npoint=32, radius=0.05, nsample=128, mlp=[128, 128, 128, 256]))
        self.SA_modules.append(PointnetSAModule(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512, 1024]))
        self.fc_layer = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(True),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.3),
        nn.ReLU(True),
        nn.Linear(1024, 1),
        nn.Sigmoid(),
        )
        # self.to(device)

    def forward(self, pc, pc_features):
        xyz = pc
        features = pc_features
        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))
