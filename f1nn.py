import numpy as np
import pandas as pd
import data as f1

TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 40
EPOCHS = 1000


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


print('')
sessions = f1.get_sessions(2023)
session_data = f1.get_filtered_session_results(sessions)
print('')

print(session_data.head())


numeric_feature_names = ['GridPosition', 'Finished']
binary_feature_names = []
categorical_feature_names = ['Abbreviation']

target_names = ['Position']


# PREPROCESSING

print('\n\n-- PREPROCESSING --\n')

features = session_data[numeric_feature_names]
targets = session_data[target_names]

train_data, test_data = model.get_data(features, targets, train_test_split=TRAIN_TEST_SPLIT, batch_size=BATCH_SIZE)


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
#pred_output = model(torch.tensor(pred_input, dtype=torch.float32).to(device))
pred_output = model.predict(nn, pred_input)
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