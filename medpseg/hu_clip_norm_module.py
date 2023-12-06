'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Proposal of module to learn normalization parameters with training

Did not work
'''
import torch
import numpy as np
from torch import nn


class CTHUClipNormModule(nn.Module):
    '''
    Clip and normalize to [0-1] if norm true, using learnable clip parameters.
    Parameters are initialized to -1024 and 600 for lung visualization.
    This did not result in any improvements and is not being used
    '''
    def __init__(self, vmin=-1024, vmax=600, norm=True):
        super().__init__()
        self.vmin = nn.Parameter(torch.tensor(vmin, dtype=torch.float32), requires_grad=True)
        self.vmax = nn.Parameter(torch.tensor(vmax, dtype=torch.float32), requires_grad=True)
        self.norm = norm
    
    def forward(self, x):
        x = torch.clip(x, self.vmin, self.vmax)
    
        if self.norm:
            x = (x - self.vmin)/(self.vmax - self.vmin)
        
        return x

    def __str__(self):
        return f"CTHUClipNormModule vmin: {self.vmin} vmax: {self.vmax} norm: {self.norm}"


class CTHUClipZNormModule(nn.Module):
    '''
    Clip and normalize to [-1 1] if norm true, using learnable clip parameters.
    Parameters are initialized to -1024 and 600 from lungmask. 
    '''
    def __init__(self, vmin=-1024, vmax=600, norm=True):
        super().__init__()
        self.vmin = nn.Parameter(torch.tensor(vmin, dtype=torch.float32), requires_grad=True)
        self.vmax = nn.Parameter(torch.tensor(vmax, dtype=torch.float32), requires_grad=True)
        self.norm = norm
    
    def forward(self, x):
        x = torch.clip(x, self.vmin, self.vmax)
    
        if self.norm:
            x = (((x - self.vmin)/(self.vmax - self.vmin))*2) - 1
        
        return x

    def __str__(self):
        return f"CTHUClipZNormModule vmin: {self.vmin} vmax: {self.vmax} norm: {self.norm} between -1 and 1"
