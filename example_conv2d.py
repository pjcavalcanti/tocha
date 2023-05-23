import tocha
import torchvision.datasets as datasets
import tocha.nn as nn
import tocha.functional as F
import matplotlib.pyplot as plt
import numpy as np

mnist = datasets.MNIST(root="./data", train=True, download=True, transform=None)

data = np.expand_dims(mnist.data.numpy(), 1) / 255.
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
    def __init__(self, image_size=(28,28), batch_size=32, num_classes=10, kernel_size=(3, 3), in_features =1 , n_hidden = 6):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.n_hidden = n_hidden
        
        self.conv1 = nn.Conv2d(in_features, n_hidden, kernel_size, bias=True)
        self.conv2 = nn.Conv2d(n_hidden, n_hidden * 2, kernel_size, bias=True)
        self.lin1 = nn.Linear(n_hidden * 2 * (image_size[0] - kernel_size[0] * 2 + 2) * (image_size[1] - kernel_size[1] * 2 + 2), num_classes, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        # print(out)
        out = F.relu(out)
        out = F.relu(self.conv2(out))
        out = out.reshape((self.batch_size, self.n_hidden * 2 * (self.image_size[0] - self.kernel_size[0] * 2 + 2) * (self.image_size[1] - self.kernel_size[1] * 2 + 2)))
        out = self.lin1(out)
        return out


model = ConvNet()
for e in range(10):
    x, y = get_batch()
    c = model(x)
    c = F.softmax(c, dim=1)
    # Check that that the following really selects the probability of the correct class
        # i = np.random.randint(0, len(x))
        # print(i)
        # print(y[i])
        # print(c[i,y[i].data.tolist()])
        # print(c[np.arange(0, c.shape[0]), idx])
    idx = y.data.tolist()
    c = c[np.arange(0, c.shape[0]), idx]
    
    
    break
