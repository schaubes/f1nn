import numpy as np
import pandas as pd
import data as f1
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
EPOCHS = 20


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


# SETUP

session = f1.get_session(2023, 1, 'R')
sessions = f1.get_sessions(2023)


# MAIN

print('----------')
print('-- F1NN --')
print('----------')

#print(session)
#print(session['results'])
#print(session.items())

session_data = f1.get_filtered_session_results(sessions)


# INPUT

print('\n\n-- INPUT --\n')

numeric_feature_names = ['GridPosition', 'Finished']
binary_feature_names = []
categorical_feature_names = ['Abbreviation']

#target_names = ['Position']
targets = session_data.pop('Position')


# PREPROCESSING

print('\n\n-- PREPROCESSING --\n')

inputs = {}
for name, column in session_data.items():
    if type(column.iloc[0]) == str:
        dtype = tf.string
    elif (name in categorical_feature_names or
            name in binary_feature_names):
        dtype = tf.int64
    else:
        dtype = tf.float32
    
    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

print(inputs)


normalizer = tf.keras.layers.Normalization(axis=-1)


preprocessed = []

# binary features
for name in binary_feature_names:
  inp = inputs[name]
  inp = inp[:, tf.newaxis]
  float_value = tf.cast(inp, tf.float32)
  preprocessed.append(float_value)

print(preprocessed)

#numeric features
numeric_inputs = {}
for name in numeric_feature_names:
  numeric_inputs[name] = inputs[name]

numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

preprocessed.append(numeric_normalized)

print(preprocessed)

# categorical features
for name in categorical_feature_names:
    vocab = sorted(set(session_data[name]))
    print(f'name: {name}')
    print(f'vocab: {vocab}\n')

    if type(vocab[0]) is str:
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
    else:
        lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

    x = inputs[name][:, tf.newaxis]
    x = lookup(x)
    preprocessed.append(x)


preprocessed_result = tf.concat(preprocessed, axis=-1)
print(preprocessed_result)

preprocessor = tf.keras.Model(inputs, preprocessed_result)

print(inputs)


# MODEL

print('\n\n-- MODEL --\n')

def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model

model = get_basic_model()


# TRAINING

print('\n\n-- TRAINING --\n')

model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)


# EVALUATION

print('\n\n-- EVALUATION --\n')

model.evaluate(inputs, targets)


# PREDICTION

print('\n\n-- PREDICTION --\n')

#predictions = model.predict(inputs.iloc[1])
#print(predictions)