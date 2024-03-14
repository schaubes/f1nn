import numpy as np
import pandas as pd
import fastf1 as f1
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 8


def get_session(year, round, session_type):
    session_data = f1.get_session(year, round, session_type)
    session_data.load(laps=False, telemetry=False, weather=False, messages=False)

    session_results = get_session_results(session_data.results)
    return session_results

    session = {
        'results': get_session_results(session_data.results),
        #'session_info': session_data.session_info,
        #'weather_data': session_data.weather_data
    }

    return session


def get_session_results(session_results):
    session_prep = pd.DataFrame(session_results, columns=['DriverNumber', 'Abbreviation', 'GridPosition', 'Position', 'ClassifiedPosition', 'Status', 'Time', 'Points'])
    #session_cols_int = ['DriverNumber', 'GridPosition', 'ClassifiedPosition', 'Position', 'Points']
    #session_cols_int = ['DriverNumber', 'GridPosition', 'Position', 'Points']
    #session_prep[session_cols_int] = session_prep[session_cols_int].astype('Int64')

    session_new = pd.DataFrame()
    session_new['Abbreviation'] = session_prep['Abbreviation']
    session_new['GridPosition'] = session_prep['GridPosition']
    session_new['Finished'] = session_prep.apply(lambda x: 1.0 if pd.to_numeric(x['ClassifiedPosition'], errors='coerce') == x['Position'] else 0.0, axis=1)
    session_new['Position'] = session_prep['Position']

    return session_new


def get_session_drivers(session_results):
    drivers = pd.unique(session_results['Abbreviation'])
    return drivers


def get_sessions(year):
    num_sessions = len(f1.get_event_schedule(year, include_testing=False).values)
    
    sessions = []
    for i in range(0, num_sessions):
        session = get_session(year, i+1, 'R')
        sessions.append(session)

    return sessions


# SETUP

sessions = get_sessions(2023)
#session = get_session(2023, 1, 'R')

print('')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

print('')


# MAIN

print('----------')
print('-- F1NN --')
print('----------')
print()

#print(session.iloc[0])
#print(session['results'])
#print(session.items())
#print(len(sessions), sessions)

session_data = pd.concat(sessions, ignore_index=True)
#session_data = sessions[0]

session_data.dropna(inplace=True)

print(session_data.head())

# INPUT

print('\n-- INPUT --\n')

numeric_feature_names = ['GridPosition', 'Finished']
binary_feature_names = []
categorical_feature_names = ['Abbreviation']
target_feature_names = ['Position']

# target = session.pop('Position')


# MODEL

print('\n-- MODEL --\n')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
#print(model)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Model methods

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def save(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Saved PyTorch Model State to {path}")

def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded PyTorch Model State from {path}")


# DATA

class F1Dataset(Dataset):
    def __init__(self, sessions):
        self.x = torch.tensor(sessions[numeric_feature_names].values, dtype=torch.float32)
        self.y = torch.tensor(sessions[target_feature_names].values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

train_test_split = 0.9
train_test_split_index = int(len(session_data) * train_test_split)

train_dataset = F1Dataset(session_data.iloc[:train_test_split_index])
test_dataset = F1Dataset(session_data.iloc[train_test_split_index:])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Input: {train_dataset[0][0].shape} | Output: {train_dataset[0][1].shape}")
print(f"Data Split: {len(session_data)} [{len(train_dataset)}:{len(test_dataset)}]")


# TRAINING

print('\n-- TRAINING --\n')

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# EVALUATION

print('\n-- EVALUATION --\n')

model.eval()

# X, y = test_dataset[0]

# with torch.no_grad():
#     pred = model(X)
#     print(f"Predicted: {pred[0]}")
#     print(f"Actual: {y[0]}")
#     print(f"Diff: {pred[0] - y[0]}")
#     print(f"Diff %: {((pred[0] - y[0]) / y[0]) * 100}")


# PREDICTION

print('\n-- PREDICTION --\n')

class ResultPrediction:
    def __init__(self, start_pos, finished, pred):
        self.start_pos = start_pos
        self.finished = finished
        self.pred = pred
        self.pos = 0
    
    def position(self, pos):
        self.pos = pos

#pred_input = [[1, 1], [2, 1]]

pred_input_starting = 20
pred_input_finishing = 20

pred_input = []
for i in range(0, pred_input_starting):
    pred_input_is_finishing = 1 if i < pred_input_finishing else 0
    pred_input.append([i+1, pred_input_is_finishing])

print(f"Input: {pred_input}")
pred_output = model(torch.tensor(pred_input, dtype=torch.float32).to(device))
#print(f"Output: {pred_output}")
#print(f"Predicted: {np.round(output.item())} ({output.item()})")

pred_results = []

for i in range(0, len(pred_output)):
    inp = pred_input[i]
    pred = pred_output[i]
    pred_results.append(ResultPrediction(inp[0], inp[1], pred.item()))

pred_results_sorted = sorted(pred_results, key=lambda x: x.pred)

for i in range(0, len(pred_results_sorted)):
    pred_result = pred_results_sorted[i]
    pred_result.position(i+1)
    print(f"{pred_result.start_pos} -> {pred_result.pos} ({pred_result.pred})")