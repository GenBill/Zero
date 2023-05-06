import torch
import torch.nn as nn
from model import Matrix
from tqdm import tqdm

device = 'cuda'

def make_math(x):
    x0 = x[:,0]
    x1 = x[:,1]
    x2 = x[:,2]
    x3 = x[:,3]
    # y = x0 + (x1>x2)*x3 + (x1-x2)/((x3+x0).abs+1e-6)
    y = x0 + (x1>x2)*x3 + (x1-x2)/(torch.abs(x3+x0)+1e-6)
    return y.unsqueeze(0)

net = Matrix(4, 1, 128, 128).to(device)
# opt = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=0.9, weight_decay=2e-5)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=[0.9,0.999], weight_decay=1e-6)

for i in range(1000):
    loss_sum = 0
    for j in tqdm(range(100)):
        x = torch.randn(16, 4).to(device)
        y = make_math(x).to(device)
        y_pred = net(x)

        loss = torch.mean(torch.abs(y-y_pred))
        loss_sum += loss.item()

        # if j%10==0:
        #     print(y[0,0].item(), y_pred[0,0].item(), loss.item())
        #     print(net.mathout.activator.param)

        opt.zero_grad()
        loss.backward()
        opt.step()
    print('epoch[',i,'] loss: ', loss_sum/1600)
