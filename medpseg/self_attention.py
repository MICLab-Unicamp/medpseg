'''
Own implementation of attention model in:

Górriz, Marc, et al. "Assessing knee OA severity with CNN attention-based end-to-end architectures." 
International conference on medical imaging with deep learning. PMLR, 2019.
'''

from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    '''
    Spatial attention module, with 1x1 convolutions, idea from
    ASSESSING KNEE OA SEVERITY WITH CNN ATTENTION-BASED END-TO-END ARCHITECTURES
    '''
    def __init__(self, in_ch, dim):
        super().__init__()
        self.first_conv = getattr(nn, f"Conv{dim}")(in_ch, in_ch//2, kernel_size=1, padding=0, stride=1, bias=False)
        self.second_conv = getattr(nn, f"Conv{dim}")(in_ch//2, in_ch//4, kernel_size=1, padding=0, stride=1, bias=False)
        self.third_conv = getattr(nn, f"Conv{dim}")(in_ch//4, 1, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        y = self.first_conv(x)
        y = F.leaky_relu(y, inplace=True)
        y = self.second_conv(y)
        y = F.leaky_relu(y, inplace=True)
        self.att = self.third_conv(y).sigmoid()
        return x*self.att