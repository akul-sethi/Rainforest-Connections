import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import time

# Paths
DATA_PATH = 'results'

# Load files

# Hyper parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SHUFFLE_SIZE = 100
EPOCHS = 5


# Spectrogram size is (1025, 2584, 1)

def generator():
    for root, dirnames, filenames in os.walk(DATA_PATH):
        for file in filenames:
            split = file.split('_')
            label = split[1].split('.')[0]
            yield np.load(os.path.join(DATA_PATH, file)), label


spec_shape = (128, 500, 1)

train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float64, tf.int8),
                                          output_shapes=(spec_shape, ()))

train_ds = train_ds.cache().shuffle(buffer_size=SHUFFLE_SIZE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

norm_layer = keras.layers.experimental.preprocessing.Normalization()
norm_layer.adapt(train_ds.map(lambda x, _: x))

model = keras.Sequential([
    layers.Input(shape=spec_shape),
    layers.Conv2D(16, (3, 3), activation='relu'),
    norm_layer,
    layers.MaxPool2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(24, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()

# Fit model
model.fit(train_ds, epochs=EPOCHS)
