import torch
import torch.nn as nn

class Matrix(nn.Module):
    def __init__(self, input_channel, output_channel, num=16, dim=128) -> None:
        super().__init__()
        self.num = num
        self.dim = dim
        self.link = []
        self.life = []
        self.pre_linear = nn.Linear(input_channel, self.dim)
        self.post_linear = nn.Linear(self.dim, output_channel)

        for i in range(self.num):
            link_temp = []
            life_temp = []
            # matrix_temp = []
            for j in range(self.num):
                link_temp.append(nn.Linear(self.dim, self.dim))
                life_temp.append(nn.Parameter(torch.rand((1))+0.5, requires_grad=True))
                # matrix_temp.append(torch.ones((1,dim), requires_grad=False))
            self.link.append(link_temp)
            self.life.append(life_temp)
            # self.matrix.append(matrix_temp)
    
    def init_matrix(self, inp):
        matrix = []
        for i in range(self.num):
            matrix.append(torch.zeros_like(inp))
        return matrix
    
    def one_step(self, inp, matrix):
        matrix2 = self.init_matrix(inp)
        for i in range(self.num):
            for j in range(self.num):
                if self.life[i][j]>0:
                    matrix2[j] += self.life[i][j] * self.link[i][j](matrix[i])
        return matrix2
    
    def update_life(self, alpha):
        for i in range(self.num):
            meanlife = 0
            for j in range(self.num):
                self.life[i][j] = torch.clamp(self.life[i][j], -1, 1)
                meanlife += self.life[i][j] / self.num
            for j in range(self.num):
                self.life[i][j] = self.life[i][j]*alpha + meanlife*(1-alpha)

    def forward(self, inp, steps=100):
        inp = self.pre_linear(inp)
        matrix = self.init_matrix(inp)
        matrix[0] = inp
        for i in range(steps):
            matrix = self.one_step(inp, matrix)
        return self.post_linear(matrix[self.num-1])