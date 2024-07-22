import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#print(f"Using {device} device")


# [ 1 grid_position, 1 finished, 30 [driver 1-to-k] ]
class NeuralNetwork(nn.Module):
    def __init__(self, feature_length, target_length):
        super().__init__()

        if (isinstance(feature_length, int) == False) or (isinstance(target_length, int) == False):
            raise ValueError("Feature and target lengths must be integers")

        if (feature_length < 1) or (target_length < 1):
            raise ValueError("Feature and target lengths must be greater than 0")

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_length, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.5),
            nn.Linear(16, target_length)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def print(self):
        print(self.linear_relu_stack)


class F1Dataset(Dataset):
    def __init__(self, features, targets):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def get_data(features, targets, train_test_split = 0.9, batch_size = 8):
    if len(features) != len(targets):
        raise ValueError("Features and targets must have the same length")

    train_test_split_index = int(len(features) * train_test_split)

    train_dataset = F1Dataset(features[:train_test_split_index], targets[:train_test_split_index])
    test_dataset = F1Dataset(features[train_test_split_index:], targets[train_test_split_index:])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Input: {train_dataset[0][0].shape} | Output: {train_dataset[0][1].shape}")
    print(f"Data Split: {len(features)} [{len(train_dataset)}:{len(test_dataset)}]")
    
    return train_dataloader, test_dataloader


def get_model(feature_length, target_length):
    model = NeuralNetwork(feature_length, target_length).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def train(dataloader, model, loss_fn, optimizer, epochs = 8):
    for t in range(epochs):
        size = len(dataloader.dataset)

        if t % 100 == 99:
            print(f"\nEpoch {t+1}\n-------------------------------")

        model.train()

        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, Y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if batch % 400 == 0:
            #     loss, current = loss.item(), (batch + 1) * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print()


def test(dataloader, model, loss_fn):
    if len(dataloader) == 0:
        return
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


def eval(model):
    model.eval()


def save(model, optimizer, path):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def load(path):
    model = NeuralNetwork(32, 2)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Loaded PyTorch Model State from {path}")
    return model


def predict(model, input):
    return model(torch.tensor(input, dtype=torch.float32).to(device))