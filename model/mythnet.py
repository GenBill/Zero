import torch
import torch.nn as nn
from .basic import Basic_Activation_Block as aBlock
from .basic import Basic_Operator_Block as oBlock
from .basic import Basic_Concat_Block as cBlock

class MythNet_demo(nn.Module):
    def __init__(self, in_nc=4, dim_nc=128, out_nc=4) -> None:
        super().__init__()
        self.premath = aBlock(in_nc, dim_nc)
        self.math0 = cBlock(dim_nc, dim_nc, dim_nc)
        self.math1 = cBlock(dim_nc, dim_nc, dim_nc)
        self.math2 = cBlock(dim_nc, dim_nc, dim_nc)
        self.math3 = cBlock(dim_nc, dim_nc, dim_nc)
        self.math4 = cBlock(dim_nc, dim_nc, dim_nc)
        self.math5 = cBlock(dim_nc, dim_nc, dim_nc)
        self.mathout = aBlock(dim_nc, out_nc)
    
    def lightforward(self, x):
        x = self.premath(x)
        x = self.math0(x)
        x = self.mathout(x)
        return x

    def forward(self, x):
        x = self.premath(x)
        
        x0 = self.math0(x)
        x1 = self.math1(x0)
        x2 = self.math2(x1)

        x = self.math3(x+x2)
        x = self.math4(x+x1)
        x = self.math5(x+x0)
        x = self.mathout(x)
        return x
