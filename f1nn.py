import numpy as np
import pandas as pd
import data as f1

TRAIN_TEST_SPLIT = 1.0
BATCH_SIZE = 20
EPOCHS = 2000


# MAIN

print('----------')
print('-- F1NN --')
print('----------')


# INPUT

print('\n\n-- INPUT --\n')

framework = 'torch'

if framework == 'tf':
    import model_tf as model
else:
    import model_torch as model


print()
sessions = f1.get_sessions_since(2023)
session_data = f1.get_filtered_session_results(sessions)
print()

print(session_data.head())


numeric_feature_names = ['GridPosition']
binary_feature_names = ['Finished']
categorical_feature_names = ['Abbreviation']

target_names = ['Position']


# PREPROCESSING

print('\n\n-- PREPROCESSING --\n')

preprocessed_features = []
preprocessed_targets = []

abbreviations = session_data['Abbreviation'].unique()
abbreviations.sort()

print(abbreviations, len(abbreviations))
print()

for index, row in session_data.iterrows():
    preprocessed_row = np.zeros(32, dtype=np.float32)

    preprocessed_row[0] = row['GridPosition']
    preprocessed_row[1] = row['Finished']

    if (row['Abbreviation'] in abbreviations):
        abbr_index = np.where(abbreviations == row['Abbreviation'])[0][0]
        preprocessed_row[2 + abbr_index] = 1

    preprocessed_features.append(preprocessed_row.tolist())

print(preprocessed_features[0:5])
print()

#features = session_data[numeric_feature_names + binary_feature_names]
targets = session_data[target_names]

train_data, test_data = model.get_data(preprocessed_features, targets.values, train_test_split=TRAIN_TEST_SPLIT, batch_size=BATCH_SIZE)


# MODEL

print('\n\n-- MODEL --\n')

nn, loss_fn, optimizer = model.get_model()

print('Model created')


# TRAINING

print('\n\n-- TRAINING --\n')

model.train(train_data, nn, loss_fn, optimizer, epochs=EPOCHS)

print('Training complete')


# EVALUATION

print('\n\n-- EVALUATION --\n')

model.eval(nn)

print('Evaluation complete')


# TESTING

print('\n\n-- TESTING --\n')

model.test(test_data, nn, loss_fn)

print('Testing complete')


# PREDICTION

print('\n\n-- PREDICTION --\n')

class ResultPrediction:
    def __init__(self, start_pos, finished, abbr, pred=0):
        self.start_pos = start_pos
        self.finished = finished
        self.abbr = abbr
        self.pred = pred
        self.pos = 0
    
    def predict(self, pred):
        self.pred = pred

    def position(self, pos):
        self.pos = pos

#grid_positions = sessions[0]['Abbreviation'].unique()
grid_positions = ['VER', 'LEC', 'PER', 'RUS', 'SAI', 'HAM', 'NOR', 'PIA', 'ALO', 'STR', 'BOT', 'RIC', 'TSU', 'GAS', 'OCO', 'MAG', 'HUL', 'ZHO', 'ALB', 'SAR']

if len(grid_positions) != 20:
    raise ValueError("Grid positions must be 20")

for i in range(0, len(grid_positions)):
    if grid_positions[i] not in abbreviations:
        raise ValueError(f"Grid position {grid_positions[i]} not found in abbreviations")

pred_inputs = []
pred_results = []

for i in range(0, len(grid_positions)):
    preprocessed_row = np.zeros(32, dtype=np.float32)

    start_pos = i + 1
    abbr = grid_positions[i]
    abbr_index = np.where(abbreviations == abbr)[0][0]
    finished = 1

    preprocessed_row[0] = start_pos
    preprocessed_row[1] = finished
    preprocessed_row[2 + abbr_index] = 1

    pred_inputs.append(preprocessed_row.tolist())
    pred_results.append(ResultPrediction(start_pos, finished, abbr))

pred_outputs = model.predict(nn, pred_inputs)

for i in range(0, len(pred_outputs)):
    pred_results[i].predict(pred_outputs[i].item())

pred_results_sorted = sorted(pred_results, key=lambda x: x.pred)

for i in range(0, len(pred_results_sorted)):
    pred_result = pred_results_sorted[i]
    pred_result.position(i+1)
    print(f"{pred_result.abbr} {pred_result.start_pos} -> {pred_result.pos} ({pred_result.pred})")



#pred_input = [[1, 1], [2, 1]]

# pred_input_starting = 20
# pred_input_finishing = 20

# pred_input = []
# for i in range(0, pred_input_starting):
#     pred_input_is_finishing = 1 if i < pred_input_finishing else 0
#     pred_input.append([i+1, pred_input_is_finishing])

# print(f"Input: {pred_input}")
# pred_output = model.predict(nn, pred_input)

# pred_results = []

# for i in range(0, len(pred_output)):
#     inp = pred_input[i]
#     pred = pred_output[i]
#     pred_results.append(ResultPrediction(inp[0], inp[1], pred.item()))

# pred_results_sorted = sorted(pred_results, key=lambda x: x.pred)

# for i in range(0, len(pred_results_sorted)):
#     pred_result = pred_results_sorted[i]
#     pred_result.position(i+1)
#     print(f"{pred_result.start_pos} -> {pred_result.pos} ({pred_result.pred})")