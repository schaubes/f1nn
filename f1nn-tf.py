import numpy as np
import pandas as pd
import data as f1
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
EPOCHS = 20


# SETUP

session = f1.get_session(2023, 1, 'R')
sessions = f1.get_sessions(2023)


# MAIN

print('----------')
print('-- F1NN --')
print('----------')
print()

#print(session)
#print(session['results'])
#print(session.items())

session_data = pd.concat(sessions, ignore_index=True)
session_data.dropna(inplace=True)

print(session_data.head())


# INPUT

print('\n-- INPUT --\n')

numeric_feature_names = ['GridPosition', 'Finished']
binary_feature_names = []
categorical_feature_names = ['Abbreviation']

target = session_data.pop('Position')


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


numeric_features = session_data[numeric_feature_names]
#print(numeric_features)
print(numeric_features.head())


# PREPROCESSING

print('\n-- PREPROCESSING --\n')

#print(tf.convert_to_tensor(numeric_features))


# preprocessed = []

# for name in binary_feature_names:
#   inp = inputs[name]
#   inp = inp[:, tf.newaxis]
#   float_value = tf.cast(inp, tf.float32)
#   preprocessed.append(float_value)

# print(preprocessed)

#print(numeric_features.values)

normalizer = tf.keras.layers.Normalization(axis=None)
normalizer.adapt(numeric_features.values)
#normalizer.adapt(dict(numeric_features.values))
#normalizer.adapt(stack_dict(dict(numeric_features.values)))


# MODEL

print('\n-- MODEL --\n')

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

print('\n-- TRAINING --\n')

model.fit(numeric_features, target, epochs=EPOCHS, batch_size=BATCH_SIZE)


# EVALUATION

print('\n-- EVALUATION --\n')

model.evaluate(numeric_features, target)


# PREDICTION

print('\n-- PREDICTION --\n')

predictions = model.predict(numeric_features)
print(predictions)