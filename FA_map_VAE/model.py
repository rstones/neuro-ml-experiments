import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class VarAutoEnc(nn.Module):

    def __init__(self):
        super(VarAutoEnc, self).__init__()

        self.channel = 512

        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//2, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)

        self.conv1 = nn.Conv3dTranpose(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3dTranspose(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3dTranspose(channel//2, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3dTranspose(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        
        def encode(self, x):
            h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
            h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
            h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
            h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
            return self.mean(h4), self.logvar(h4) # reparameterization happens here
        
        def reparameterize(self, mu, logvar):
            pass

        def decode(self, z):
            pass

        def forward(self, x):
            pass

