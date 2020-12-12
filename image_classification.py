import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import time

# Paths
FEATURES_PATH = 'train_tp_specs.npy'
LABELS_PATH = 'train_tp_labels.npy'

# Load files
labels_file = np.load(LABELS_PATH, mmap_mode='r')
features_file = np.load(FEATURES_PATH, mmap_mode='r')

# Hyper parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SHUFFLE_SIZE = 100
EPOCHS = 5

spec_shape = (features_file.shape[1], features_file.shape[2], 1)


# Spectrogram size is (1025, 2584, 1)

def generator():
    for i, f in enumerate(features_file):
        yield f, labels_file[i]


def normalize(*args):
    new_spec = (args[0] + 73.864) / (52.06553 + 73.864)
    new_spec = tf.reshape(new_spec, [BATCH_SIZE, spec_shape[0], spec_shape[1], spec_shape[2]])
    return new_spec, args[1]


# Max value is 52.06553
# Min value is -73.864


train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float64, tf.int32),
                                          output_shapes=((1025, 2584), ()))
train_ds = train_ds.shuffle(buffer_size=SHUFFLE_SIZE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=spec_shape),
    layers.MaxPool2D(pool_size=(15, 15)),
    layers.Conv2D(16, (4, 4), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, (4, 4), activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(24, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()

# Fit model
history = model.fit(train_ds, epochs=EPOCHS)
