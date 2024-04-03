import numpy as np
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

target_names = ['Position', 'Finished']


# PREPROCESSING

print('\n\n-- PREPROCESSING --\n')

preprocessed_features = []
preprocessed_targets = []

abbreviations = session_data['Abbreviation'].unique()
abbreviations.sort()

print('Unique drivers', len(abbreviations), abbreviations)
print()

# Preprocess features

for index, row in session_data.iterrows():
    preprocessed_row = np.zeros(32, dtype=np.float32)

    preprocessed_row[0] = row['GridPosition']
    #preprocessed_row[1] = row['Finished']

    if (row['Abbreviation'] in abbreviations):
        abbr_index = np.where(abbreviations == row['Abbreviation'])[0][0]
        preprocessed_row[2 + abbr_index] = 1

    preprocessed_features.append(preprocessed_row.tolist())

print(preprocessed_features[0])
print()

# Preprocess targets

for index, row in session_data.iterrows():
    preprocessed_row = np.zeros(2, dtype=np.float32)

    preprocessed_row[0] = row['Position']
    preprocessed_row[1] = row['Finished']

    preprocessed_targets.append(preprocessed_row.tolist())

print(preprocessed_targets[0])
print()


# Split data

train_data, test_data = model.get_data(preprocessed_features, preprocessed_targets, train_test_split=TRAIN_TEST_SPLIT, batch_size=BATCH_SIZE)


# MODEL

print('\n\n-- MODEL --\n')

feature_length = len(preprocessed_features[0])
target_length = len(preprocessed_targets[0])

nn, loss_fn, optimizer = model.get_model(feature_length, target_length)

nn.print()
print()

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
    def __init__(self, start_pos, abbr, pred=0):
        self.start_pos = start_pos
        self.abbr = abbr
        self.pred = pred
        self.finished = 0
        self.pos = 0
    
    def setPrediction(self, pred):
        self.pred = pred

    def setFinished(self, finished):
        self.finished = finished

    def setPosition(self, pos):
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
    # finished = 1

    preprocessed_row[0] = start_pos
    # preprocessed_row[1] = finished
    preprocessed_row[2 + abbr_index] = 1

    pred_inputs.append(preprocessed_row.tolist())
    pred_results.append(ResultPrediction(start_pos, abbr))

pred_outputs = model.predict(nn, pred_inputs)

for i in range(0, len(pred_outputs)):
    pred_results[i].setPrediction(pred_outputs[i, 0].item())
    pred_results[i].setFinished(pred_outputs[i, 1].item())

# Print predictions by position

print("Race predictions\n")

pred_results_sorted = sorted(pred_results, key=lambda x: x.pred)

for i in range(0, len(pred_results_sorted)):
    pred_result = pred_results_sorted[i]
    pred_result.setPosition(i+1)
    print(f"{pred_result.abbr} {pred_result.start_pos} -> {pred_result.pos} ({pred_result.pred}, {pred_result.finished})")

print()

# Print predictions by finished

print("Finishing predictions\n")

pred_results_finished_sorted = sorted(pred_results, key=lambda x: x.finished)

for i in range(0, len(pred_results_finished_sorted)):
    pred_result = pred_results_finished_sorted[i]
    print(f"{pred_result.abbr} {pred_result.finished} ({pred_result.pos} -> {pred_result.pred})")

print()