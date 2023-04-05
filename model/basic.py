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
        # sin2 = y*torch.sin(x)
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
        self.param = torch.rand((len(self.Operator_Set), self.N), grad=True)
        self.softmax = nn.Softmax(?, dim=1)
    
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
        return torch.log(torch.abs(x)+1e-8)

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

# Activation_Set = [inv(), log(), exp(), sin(), relu()]
class Basic_Activation(nn.Module):
    def __init__(self, num) -> None:
        super().__init__()
        self.Activation_Set = [inv(), log(), exp(), sin(), relu()]
        self.N = num
        self.param = torch.rand((len(self.Activation_Set), self.N), grad=True)
        self.softmax = nn.Softmax(?, dim=1)
    
    def forward(self, x):
        soft_param = self.softmax(self.param)
        ret = 0
        for i, activate in enumerate(self.Activation_Set):
            for j in range(self.N):
                ret += activate(x) * soft_param[i, j]
        return ret
