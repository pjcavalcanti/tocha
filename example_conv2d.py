import numpy as np
import tocha
import tocha.nn as nn
from tocha.functional import relu, sigmoid, flatten
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class LeNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6, (5, 5), bias=True)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), bias=True)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = flatten(x)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the MNIST dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(
    "mnist_data/", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(
    "mnist_data/", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
model = LeNet()

# Training loop
for epoch in range(10):  # run for 10 epochs
    for idx, (images, labels) in enumerate(train_loader):
        # Convert PyTorch Tensor to Numpy array and then to Tocha Tensor
        images = tocha.tensor(images.numpy())
        labels = tocha.tensor(labels.numpy())

        outputs = model(images)
        loss = ((outputs - labels) ** 2).mean()

        loss.backward()  # backward pass

        # Update weights
        for p in model.parameters():
            p.data -= 0.01 * p.grad.data  # we are using a learning rate of 0.01
            p.zero_grad()  # reset gradients

print("Finished Training")
