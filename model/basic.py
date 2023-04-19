import torch
import torch.nn as nn

# Operator
class add(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        return x + y

class mul(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        return x*y

class div1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        div1 = x/(torch.abs(y)+1e-8)
        return div1

class div2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        div2 = y/(torch.abs(x)+1e-8)
        return div2

class mul_sin1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        sin1 = x*torch.sin(y)
        return sin1

class mul_sin2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        sin2 = y*torch.sin(x)
        return sin2

class Basic_Operator(nn.Module):
    def __init__(self, num) -> None:
        super().__init__()
        self.Operator_Set = [add(), mul(), div1(), div2(), mul_sin1(), mul_sin2()]
        self.N = num
        param = torch.rand((len(self.Operator_Set), self.N), requires_grad=True)
        self.param = nn.Parameter(param, requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x, y):
        soft_param = self.softmax(self.param)
        ret = 0
        for i, operator in enumerate(self.Operator_Set):
            for j in range(self.N):
                ret += operator(x,y) * soft_param[i, j]
        return ret

# Activation
class inv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return 1/(torch.abs(x)+1e-8)

class log(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return torch.log(torch.abs(x)+1)

class exp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return torch.exp(x)

class sin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return torch.sin(x)

class relu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(x)

class iden(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return x

# Activation_Set = [inv(), log(), exp(), sin(), relu(), iden()]
class Basic_Activation(nn.Module):
    def __init__(self, num) -> None:
        super().__init__()
        self.Activation_Set = [inv(), log(), exp(), sin(), relu(), iden()]
        self.N = num
        param = torch.rand((len(self.Activation_Set), self.N), requires_grad=True)
        self.param = nn.Parameter(param, requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        soft_param = self.softmax(self.param)
        ret = 0
        for i, activate in enumerate(self.Activation_Set):
            for j in range(self.N):
                ret += activate(x) * soft_param[i, j]
        return ret

class Basic_Operator_Block(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_channel, output_channel, bias=True)
        self.linear2 = nn.Linear(input_channel, output_channel, bias=True)
        self.operator = Basic_Operator(output_channel)
    
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        y = self.operator(x1, x2)
        return y

class Basic_Activation_Block(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(input_channel, affine=True)
        self.linear = nn.Linear(input_channel, output_channel, bias=True)
        self.activator = Basic_Activation(output_channel)
    
    def forward(self, x):
        y = self.bn(x)
        y = self.linear(y)
        y = self.activator(y)
        return y

class Basic_Concat_Block(nn.Module):
    def __init__(self, input_channel, mid_channel, output_channel) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_channel, affine=True)
        self.linear1 = nn.Linear(input_channel, mid_channel, bias=True)
        self.linear2 = nn.Linear(input_channel, mid_channel, bias=True)
        self.operator = Basic_Operator(mid_channel)
        self.bn2 = nn.BatchNorm1d(mid_channel, affine=True)
        self.linear3 = nn.Linear(mid_channel*2, output_channel, bias=True)
        self.activator = Basic_Activation(output_channel)
    
    def forward(self, x):
        x = self.bn1(x)
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        y = self.operator(x1, x2)
        y = self.bn2(y)
        y = self.linear3(torch.cat((x,y), dim=1))
        y = self.activator(y)
        return y
