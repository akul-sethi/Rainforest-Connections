import numpy as np
import librosa as lb
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
import time
import sys

b = np.arange(1000).reshape(10, 10, 10)
a = tf.data.Dataset.from_tensor_slices(b)
a.map(lambda x: resize(x, (5, 5)))
print(list(a.take(5).as_numpy_iterator()))
