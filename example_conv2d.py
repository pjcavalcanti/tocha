import tocha
import torchvision.datasets as datasets
import tocha.nn as nn
import tocha.functional as F
import matplotlib.pyplot as plt
import numpy as np

mnist = datasets.MNIST(root="./data", train=True, download=True, transform=None)

data = np.expand_dims(mnist.data.numpy(), 1)
labels = mnist.targets.numpy()

train_split = int(0.8 * len(data))
x_train, y_train = data[:train_split], labels[:train_split]
x_test, y_test = data[train_split:], labels[train_split:]

# shuffle data
idx = np.random.permutation(len(x_train))
x_train, y_train = x_train[idx], y_train[idx]
idx = np.random.permutation(len(x_test))
x_test, y_test = x_test[idx], y_test[idx]

x_train, y_train = tocha.tensor(x_train), tocha.tensor(y_train)
x_test, y_test = tocha.tensor(x_test), tocha.tensor(y_test)


def get_batch(batch_size=32, split="train"):
    if split == "train":
        x, y = x_train, y_train
    else:
        x, y = x_test, y_test
    idx = np.random.randint(0, len(x), batch_size)
    return x[idx], y[idx]


xb, yb = get_batch()


class ConvNet(nn.Module):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 3, (3, 3), bias=True)
        self.conv2 = nn.Conv2d(3, 6, (3, 3), bias=True)
        self.lin1 = nn.Linear(6 * 24 * 24, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.reshape((self.batch_size, 6 * 24 * 24))
        out = self.lin1(out)
        return out


model = ConvNet()
for e in range(10):
    x, y = get_batch()
    c = model(x)
    print(e)
