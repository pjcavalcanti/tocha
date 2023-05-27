import tocha
import torchvision.datasets as datasets
import tocha.nn as nn
import tocha.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

mnist = datasets.MNIST(root="./data", train=True, download=True, transform=None)

data = np.expand_dims(mnist.data.numpy(), 1) / 255.0
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


class ConvNet(nn.Module):
    def __init__(
        self,
        image_size=(28, 28),
        batch_size=32,
        num_classes=10,
        kernel_size=(3, 3),
        in_features=1,
        n_hidden=6,
    ):
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.n_hidden = n_hidden

        self.conv1 = nn.Conv2d(in_features, n_hidden, kernel_size, bias=True)
        self.conv2 = nn.Conv2d(n_hidden, n_hidden * 2, kernel_size, bias=True)
        self.lin1 = nn.Linear(
            n_hidden
            * 2
            * (image_size[0] - kernel_size[0] * 2 + 2)
            * (image_size[1] - kernel_size[1] * 2 + 2),
            num_classes,
            bias=True,
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.reshape(
            (
                self.batch_size,
                self.n_hidden
                * 2
                * (self.image_size[0] - self.kernel_size[0] * 2 + 2)
                * (self.image_size[1] - self.kernel_size[1] * 2 + 2),
            )
        )
        out = self.lin1(out)
        return out


model = ConvNet()
losses = []
val_losses = []
n_epochs = 2000
for e in tqdm(range(n_epochs)):
    x, y = get_batch()
    c = model(x)

    loss = F.cross_entropy(c, y)
    losses.append(loss.data)

    xval, yval = get_batch(split="test")
    val_loss = F.cross_entropy(model(xval), yval)
    val_losses.append(val_loss.data)

    model.zero_grad()
    loss.backward()
    lr = 0.01
    if e > 50:
        lr = 0.001
    if e > 1000:
        lr = 0.00001
    for p in model.parameters():
        p.data -= lr * p.grad.data  # type: ignore

final_loss = np.mean(losses[-20:])
final_val_loss = np.mean(val_losses[-20:])

print(f"Final train loss: {final_loss}")
print(f"Final val loss: {final_val_loss}")

plt.plot(losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.legend()
plt.show()
plt.cla()

x, y = get_batch(batch_size=32, split="test")
pred = model(x)
pred_labels = pred.data.argmax(axis=1)
print(pred_labels)

fig = plt.figure(figsize=(9, 9))
for i in range(32):
    plt.subplot(8, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x[i].data.squeeze(), cmap=plt.cm.binary)
    plt.title(f"Actual: {y[i].data} Predicted: {pred_labels[i]}")
plt.tight_layout()
plt.title(f"Final train loss: {final_loss} Final val loss: {final_val_loss}")
plt.show()
