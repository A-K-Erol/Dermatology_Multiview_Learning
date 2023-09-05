import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas


class DermDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        x = torch.tensor(self.dataframe.iloc[index, :-1], dtype=torch.float)
        y = torch.tensor(self.dataframe.iloc[index, -1], dtype=torch.long)
        return x, y


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            # highest acc results: 1xlinear - 93.8%
            #                      2xlinear - 92.3%
            #                      3x linear: 81.5
            #                      lin+relu - 76.9%
            #                      lin relu lin - 83.1%

            nn.Linear(34, 34),
            nn.Linear(34, 34)

        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def neural_net():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = NeuralNetwork().to(device)
    print(model)

    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_data = DataLoader(DermDataset(pandas.read_csv("dermTrain.csv", dtype=np.int32)), batch_size=32, shuffle=True)
    test_data = DataLoader(DermDataset(pandas.read_csv("dermTest.csv", dtype=np.int32)), batch_size=32, shuffle=True)
    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_data, model, loss_fn, optimizer)
        test_loop(test_data, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    neural_net()
