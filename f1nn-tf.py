import numpy as np
import pandas as pd
import fastf1 as f1
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2


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
    session_new['Finished'] = session_prep.apply(lambda x: 1 if pd.to_numeric(x['ClassifiedPosition'], errors='coerce') == x['Position'] else 0, axis=1)
    session_new['Position'] = session_prep['Position']

    return session_new


def get_session_drivers(session_results):
    drivers = pd.unique(session_results['Abbreviation'])
    return drivers


# SETUP

session = get_session(2023, 1, 'R')

# MAIN

print('----------')
print('-- F1NN --')
print('----------')
print()

print(session)
#print(session['results'])
#print(session.items())

# INPUT

print('\n-- INPUT --\n')

numeric_feature_names = ['GridPosition', 'Finished']
binary_feature_names = []
categorical_feature_names = ['Abbreviation']

target = session.pop('Position')

# inputs = {}
# for name, column in session.items():
#   if type(column.iloc[0]) == str:
#     dtype = tf.string
#   elif (name in categorical_feature_names or
#         name in binary_feature_names):
#     dtype = tf.int64
#   else:
#     dtype = tf.float32
  
#   inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

# print(inputs)

numeric_features = session[numeric_feature_names]
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


normalizer = tf.keras.layers.Normalization(axis=-1)
#input_shape = numeric_features.shape
#normalizer.adapt(numeric_features)
#normalizer.adapt(stack_dict(dict(numeric_features)))
print(normalizer(numeric_features.iloc[:5]))


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
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)
