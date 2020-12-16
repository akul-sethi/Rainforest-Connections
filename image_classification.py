import numpy as np
import librosa as lb
import librosa.display
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import math
import time

# Paths
TRAIN_DATA_PATH = 'results'
TEST_DATA_PATH = 'archive'

# Hyper parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SHUFFLE_SIZE = 100
EPOCHS = 1


# Spectrogram size is (128, 500, 1)
# Number of train files is about 32*36
# Number of test files is 1992

def train_generator(start, end):
    for root, dirnames, filenames in os.walk(TRAIN_DATA_PATH):
        for file in filenames[start:end]:
            split = file.split('_')
            label = split[1].split('.')[0]
            yield np.load(os.path.join(TRAIN_DATA_PATH, file)), label


def test_generator():
    specs = []
    for root, dirnames, filenames in os.walk(TEST_DATA_PATH):
        for file in filenames:
            specs.append(np.load(os.path.join(TEST_DATA_PATH, file)))
    return np.concatenate(specs)


test_ds = test_generator()

spec_shape = (128, 500, 1)

sample_size = 10
stride_size = 10
num_specs_in_sample = math.floor((60 - sample_size) / stride_size) + 1

train_ds = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float64, tf.int8),
                                          output_shapes=(spec_shape, ()), args=(tf.constant(0), tf.constant(800)))

train_ds = train_ds.cache().shuffle(buffer_size=SHUFFLE_SIZE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float64, tf.int8),
                                        output_shapes=(spec_shape, ()), args=(tf.constant(800), tf.constant(-1)))
val_ds = val_ds.cache().batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

norm_layer = keras.layers.experimental.preprocessing.Normalization()
norm_layer.adapt(train_ds.map(lambda x, _: x))

model = keras.Sequential([
    layers.Input(shape=spec_shape),
    norm_layer,
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(24, activation='softmax')
])

model.compile(keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

# Fit model
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
# model.load_weights('saved_model/')
predictions = model.predict(test_ds)
print(predictions)
print(predictions.shape)

predictions = np.reshape(predictions, (-1, num_specs_in_sample, 24))
print(predictions)
print('Done with submissions')
submission = np.max(predictions, axis=1)
print(submission)

columns = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12',
           's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23']

dataframe = pd.DataFrame(submission, columns=columns)

for dirname, _, filelist in os.walk(TEST_DATA_PATH):
    files = []
    for file in filelist:
        files.append(file.split('.')[0])
    dataframe.insert(0, 'recording_id', files)

dataframe.to_csv('submission.csv', index=False)
