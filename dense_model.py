import librosa as lb
import numpy as np
import matplotlib as plt
import pandas as pd
import librosa.display
import tensorflow.keras as keras
from tensorflow.keras import layers

TRAINING_DATA_PATH = 'train_tp_specs.npy'
LABELS_PATH = 'train_tp_labels.npy'

#Training parameters
learning_rate = 0.0001

training_data = np.load(TRAINING_DATA_PATH)
labels = np.load(LABELS_PATH)


model = keras.Sequential([
    layers.Flatten(input_shape=(training_data.shape[1], training_data.shape[2])),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(24, activation='softmax'),
])


optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()
model.fit(training_data, labels, batch_size=32, epochs=50)