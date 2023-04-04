import torch
import torch.nn as nn

# Operator
class add(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.ones((1,1,1,1), grad=True)
        self.b = torch.ones((1,1,1,1), grad=True)
    def forward(self, x, y):
        return self.a*x + self.b*y

class mul(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x, y):
        return x*y

class div(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.ones((1,1,1,1), grad=True)*0.5
    def forward(self, x, y):
        div1 = self.a * x/(torch.abs(y)+1e-8)
        div2 = (1-self.a) * y/(torch.abs(x)+1e-8)
        return div1 + div2

class Basic_Operator(nn.Module):
    def __init__(self, num) -> None:
        super().__init__()
        self.Operator_Set = [add(), mul(), div()]
        self.N = num
        self.param = torch.rand((len(self.Operator_Set), self.N), grad=True)
        self.softmax = nn.softmax(?, dim=1)
    
    def forward(self, x, y):
        soft_param = self.softmax(self.param)
        ret = 0
        for i, operator in enumerate(self.Activation_Set):
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
        self.softmax = nn.softmax(?, dim=1)
    
    def forward(self, x):
        soft_param = self.softmax(self.param)
        ret = 0
        for i, activate in enumerate(self.Activation_Set):
            for j in range(self.N):
                ret += activate(x) * soft_param[i, j]
        return ret
