import torch
import numpy as np
import matplotlib.pyplot as plt
# from torch.optim.optimizer import _params_t

np.random.seed(0)
torch.manual_seed(0)

num_samples = 500

# Inner circle
inner_radius = 2
inner_angle = np.random.uniform(0, 2 * np.pi, num_samples // 2)
inner_x = inner_radius * np.cos(inner_angle)
inner_y = inner_radius * np.sin(inner_angle)
X_inner = np.column_stack([inner_x, inner_y])
Y_inner = np.zeros((num_samples // 2, 1))

# Outer circle
outer_radius = 5
outer_angle = np.random.uniform(0, 2 * np.pi, num_samples // 2)
outer_x = outer_radius * np.cos(outer_angle)
outer_y = outer_radius * np.sin(outer_angle)
X_outer = np.column_stack([outer_x, outer_y])
Y_outer = np.ones((num_samples // 2, 1))

# Concatenate inner and outer circle
X = np.vstack([X_inner, X_outer]).astype(np.float64)
Y = np.vstack([Y_inner, Y_outer]).astype(np.float64)

# add Gaussian noise
X += np.random.normal(0, 0.3, size=X.shape)

X = X.astype(np.float64)
Y = Y.astype(np.float64)

# go to tensors
X = torch.tensor(X)
Y = torch.tensor(Y)

class MLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, dtype=torch.float64):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hidden, dtype=dtype)
        self.fc2 = torch.nn.Linear(n_hidden, n_out, dtype=dtype)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class SGD:
    def __init__(self, params, lr, momentum=0):
        self.momentum = momentum
        self.lr = lr
        self.params = list(params)
        self.v = [torch.zeros(p.shape) for p in self.params]
    def step(self):
        for i, p in enumerate(self.params):
            # if p.grad is None:
            #     continue
            
            # if self.momentum == 0:
            p.data -= self.lr * p.grad.data
            # else:
            #     if i == 0:
            #         self.v = torch.zeros_like(p.data)
            #     self.v[i] = self.momentum * self.v[i] + (1 - self.momentum) * p.grad.data
            #     p.data -= self.lr * self.v
    def zero_grad(self):
        for p in self.params:
            p.grad = torch.zeros_like(p.data)
                 
X1 = X.clone()
X2 = X.clone()
Y1 = Y.clone()
Y2 = Y.clone()

m1 = MLP(2, 3, 1, dtype=torch.float64)
m2 = MLP(2, 3, 1, dtype=torch.float64)

for p1, p2 in zip(m1.parameters(), m2.parameters()):
    p2.data = p1.data.clone()

momentum = 0
lr = 0.1
opt1 = torch.optim.SGD(m1.parameters(), lr=lr, momentum=momentum)               
opt2 = SGD(m2.parameters(), lr=lr * 1, momentum=momentum)

losses1 = []
losses2 = []
for e in range(1000):
    idx = torch.randint(0, X1.shape[0], (32,))
    
    x1 = X1[idx]
    x2 = X2[idx]
    
    loss1 = torch.nn.functional.mse_loss(m1(x1), Y1[idx])
    loss2 = torch.nn.functional.mse_loss(m2(x2), Y2[idx])
    
    opt1.zero_grad()
    loss1.backward()
    opt1.step()
    
    opt2.zero_grad()
    loss2.backward()
    opt2.step()
    
    # print(loss1.item(), loss2.item())
    losses1.append(loss1.item())
    losses2.append(loss2.item())

plt.plot(losses1, label='PyTorch')
plt.plot(losses2, label='SGD')
plt.legend()
plt.show()