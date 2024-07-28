import argparse
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

#print('\n\n-- INPUT --\n')

parser = argparse.ArgumentParser()
parser.add_argument("mode")
args = parser.parse_args()

mode = args.mode


# MODEL

framework = 'torch'
if framework == 'tf':
    import model_tf as model
else:
    import model_torch as model


# HELPER FUNCTIONS

def get_driver_data(session_data):
    abbreviations = session_data['Abbreviation'].unique()
    abbreviations.sort()

    driver_data = {}

    for abbr in abbreviations:
        driver_sessions = session_data[session_data['Abbreviation'] == abbr]
        driver_finished = driver_sessions['Finished'].sum()
        driver_data[abbr] = driver_finished / len(driver_sessions)

    return driver_data, abbreviations


# DATA MODE

def data_mode():
    sessions = f1.get_sessions_since(2023)
    session_data = f1.get_filtered_session_results(sessions)
    print()
    print(session_data.head())
    f1.save(session_data, 'data/session_data.csv')
    print()
    print('Data saved')


# TRAINING MODE

def train_mode():
    session_data = f1.load('data/session_data.csv')

    if session_data is None:
        print('No session data found. Run data mode first.')
        exit(1)

    numeric_feature_names = ['GridPosition', 'Recency']
    binary_feature_names = ['Finished']
    categorical_feature_names = ['Abbreviation']

    target_names = ['Position', 'Finished']


    # PREPROCESSING

    print('\n\n-- PREPROCESSING --\n')

    preprocessed_features = []
    preprocessed_targets = []

    driver_data, abbreviations = get_driver_data(session_data)

    # Preprocess features

    print('Feature example\n')

    for index, row in session_data.iterrows():
        preprocessed_row = np.zeros(32, dtype=np.float32)

        preprocessed_row[0] = row['GridPosition']
        # preprocessed_row[1] = row['Finished']
        preprocessed_row[1] = row['Recency']
        preprocessed_row[2] = driver_data[row['Abbreviation']]

        if (row['Abbreviation'] in abbreviations):
            abbr_index = np.where(abbreviations == row['Abbreviation'])[0][0]
            preprocessed_row[3 + abbr_index] = 1

        preprocessed_features.append(preprocessed_row.tolist())

    print(preprocessed_features[0])
    print()

    # Preprocess targets

    print('Target example\n')

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

    # SAVE MODEL

    print('\n\n-- SAVING --\n')

    model.save(nn, optimizer, 'data/model.pth')


# PREDICTION MODE

def predict_mode():
    session_data = f1.load('data/session_data.csv')

    if session_data is None:
        print('No session data found. Run data mode first.')
        exit(1)

    nn = model.load('data/model.pth')

    if nn is None:
        print('No model found. Run training mode first.')
        exit(1)

    driver_data, abbreviations = get_driver_data(session_data)


    # PREDICTION

    class ResultPrediction:
        def __init__(self, start_pos, recency, abbr, finishing_ratio, pred=0):
            self.start_pos = start_pos
            self.recency = recency
            self.abbr = abbr
            self.finishing_ratio = finishing_ratio
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
    grid_file = open('grid.txt', 'r')
    grid_positions = [line[:-1] if line[-1] == '\n' else line for line in grid_file.readlines()]

    # if len(grid_positions) != 20:
    #     raise ValueError("Grid positions must be 20")

    for i in range(0, len(grid_positions)):
        if grid_positions[i] not in abbreviations:
            raise ValueError(f"Grid position {grid_positions[i]} not found in abbreviations")

    pred_inputs = []
    pred_results = []

    for i in range(0, len(grid_positions)):
        preprocessed_row = np.zeros(32, dtype=np.float32)

        start_pos = i + 1
        recency = 1.0
        abbr = grid_positions[i]
        abbr_index = np.where(abbreviations == abbr)[0][0]
        # finished = 1

        preprocessed_row[0] = start_pos # Start position
        # preprocessed_row[1] = finished
        preprocessed_row[1] = recency # Recency
        preprocessed_row[2] = driver_data[abbr]
        preprocessed_row[3 + abbr_index] = 1 # Abbreviation

        pred_inputs.append(preprocessed_row.tolist())
        pred_results.append(ResultPrediction(start_pos, recency, abbr, driver_data[abbr]))

    pred_outputs = model.predict(nn, pred_inputs)

    for i in range(0, len(pred_outputs)):
        pred_results[i].setPrediction(pred_outputs[i, 0].item())
        pred_results[i].setFinished(pred_outputs[i, 1].item())

    # Print predictions by position

    print('\nRace predictions\n')

    pred_results_sorted = sorted(pred_results, key=lambda x: x.pred)

    for i in range(0, len(pred_results_sorted)):
        pred_result = pred_results_sorted[i]
        pred_result.setPosition(i+1)
        print(f"{pred_result.abbr} {pred_result.start_pos:>2d} -> {pred_result.pos:>2d} ({pred_result.pred:>0.7f})")
        #print(f"{pred_result.abbr} {pred_result.start_pos:>2d} -> {pred_result.pos:>2d} ({pred_result.pred:>0.7f}, {pred_result.finished:>0.7f})")

    print()

    # Print predictions by finished

    # print('Finishing predictions\n')

    # pred_results_finished_sorted = sorted(pred_results, key=lambda x: x.finished)

    # for i in range(0, len(pred_results_finished_sorted)):
    #     pred_result = pred_results_finished_sorted[i]
    #     print(f"{pred_result.abbr} {pred_result.finished:>0.7f} ({pred_result.pos:>2d} -> {pred_result.pred:>0.7f})")

    # print()

    # Print finishing analysis

    print('Finishing analysis\n')

    driver_data_sorted = {k: v for k, v in sorted(driver_data.items(), key=lambda item: item[1])}

    for abbr, ratio in driver_data_sorted.items():
        pred_result = next((x for x in pred_results if x.abbr == abbr), None)
        if pred_result is None:
            continue

        print(f"{pred_result.abbr} {ratio:>0.2f}")


# MODE

if mode == 'all':
    print('\n\n-- DATA MODE --\n')
    data_mode()
    print('\n\n-- TRAINING MODE --\n')
    train_mode()
    print('\n\n-- PREDICTION MODE --\n')
    predict_mode()
elif mode == 'main':
    print('\n\n-- TRAINING MODE --\n')
    train_mode()
    print('\n\n-- PREDICTION MODE --\n')
    predict_mode()
elif mode == 'data':
    print('\n\n-- DATA MODE --\n')
    data_mode()
elif mode == 'train':
    print('\n\n-- TRAINING MODE --\n')
    train_mode()
elif mode == 'predict' or mode == 'pred':
    print('\n\n-- PREDICTION MODE --\n')
    predict_mode()
else:
    print('\nInvalid mode: ' + mode)
    exit(1)

exit(0)