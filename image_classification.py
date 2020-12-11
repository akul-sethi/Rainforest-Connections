import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import time

FEATURES_PATH = 'train_tp_specs.npy'
LABELS_PATH = 'train_tp_labels.npy'

# Hyper parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 5


# Data Processing methods
def load_batch(batch_size):
    labels_file = np.load(LABELS_PATH, mmap_mode='r').tolist()
    features_file = np.load(FEATURES_PATH, mmap_mode='r')
    random_indexes = np.random.rand(batch_size) * (features_file.shape[0] - 1)
    random_indexes = np.around(random_indexes).astype('int64')
    num_batches = round(features_file.shape[0] / batch_size)
    for n in range(num_batches):
        batch_features = []
        batch_labels = []
        for index in random_indexes:
            shape = features_file[index].shape
            batch_features.append(features_file[index].reshape(shape[0], shape[1], 1))
            batch_labels.append(labels_file[index])

        yield np.stack(batch_features), np.stack(batch_labels)


def normalize(spectogram):
    result = (spectogram + 73.864) / (52.06553 + 73.864)
    return result


a = np.load(FEATURES_PATH, mmap_mode='r')[1]
spec_shape = (a.shape[0], a.shape[1], 1)

# Max value is 52.06553
# Min value is -73.864
# Spectrogram size is (1025, 2584, 1)


model = keras.Sequential([
    layers.Input(shape=spec_shape),
    layers.Conv2D(64, (4, 4), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, (4, 4), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, (4, 4), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(24, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()

# Fit model
for e in range(EPOCHS):
    for features, labels in load_batch(BATCH_SIZE):
        features = normalize(features)
        model.fit(features, labels, batch_size=BATCH_SIZE, epochs=1)



