'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.


Inspired by insights from COPLE-Net
'''
import torch
from torch import nn
from torch.nn import L1Loss
from monai.losses import DiceLoss, GeneralizedDiceLoss


class CombinedLoss(nn.Module):
    '''
    This class performs different loss combinations
    '''
    def __init__(self, include_background, cross_entropy=False, gdl=False, soft_circulatory=False):
        super().__init__()
        self.include_background = include_background
        self.gdl = gdl
        self.cross_entropy = cross_entropy
        self.soft_circulatory = soft_circulatory

        if self.gdl:
            dice_str = "MONAI GDL"
            self.dice = GeneralizedDiceLoss(include_background=self.include_background)
        else:
            dice_str = "MONAI DiceLoss"
            self.dice = DiceLoss(include_background=self.include_background)

        if self.cross_entropy:
            self.cross_entropy_loss = nn.NLLLoss()
            self.initialization_string = f"CombinedLoss combining {dice_str} and torch NLLLoss, include background: {self.include_background}"
            print(f"WARNING: Cross Entropy always is including background! {self.include_background} is only affecting {dice_str}")
        else:
            self.l1loss = L1Loss()
            self.initialization_string = f"CombinedLoss combining {dice_str} and torch L1Loss, include background: {self.include_background}"
        
        print(self.initialization_string)
        
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        nchannels = y_hat.shape[1]
        assert nchannels == y.shape[1], f"Loss needs equal number of channels between input and target: {nchannels}/{y.shape[1]}"

        dice = self.dice(y_hat, y)
        if nchannels == 1 or self.include_background:
            if self.cross_entropy:
                # https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net
                second_loss = self.cross_entropy_loss(torch.log(y_hat + 1e-20), y.argmax(dim=1))
            else:
                second_loss = self.l1loss(y_hat, y)
        else:
            if self.cross_entropy:
                # https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net
                second_loss = self.cross_entropy_loss(torch.log(y_hat + 1e-20), y.argmax(dim=1))
            else:
                second_loss = self.l1loss(y_hat[:, 1:], y[:, 1:])
        
        return dice + second_loss
    
    def __str__(self):
        return self.initialization_string
