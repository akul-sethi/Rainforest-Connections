# packages
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa as lb
from tqdm import tqdm
from skimage.transform import resize
import librosa.display
import matplotlib.pyplot as plt
import os

TRAIN_PATH = "rfcx-species-audio-detection/train/"
TEST_DATA_PATH = 'rfcx-species-audio-detection/test'

train_tp = pd.read_csv("rfcx-species-audio-detection/train_tp.csv", delimiter=',')
train_fp = pd.read_csv("rfcx-species-audio-detection/train_fp.csv", delimiter=',')


# recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max
# Minimum frequency is 93.75
# Maximum frequency is 13687.5
# Duration of largest sample is 7.923900000000003


def prep_mel_spec_files(train_path, sr=48000, n_fft=2048, hop_length=512, duration=10, new_shape=(128, 500), power=1.5):
    min_fr = 93.75 * 0.9
    max_fr = 13687.5 * 1.1
    duration = duration * sr
    for index, row in tqdm(train_tp.iterrows()):
        signal, sr = lb.load(os.path.join(train_path, row['recording_id']) + '.flac', sr=None)

        t_min = round(row['t_min'] * sr)
        t_max = round(row['t_max'] * sr)

        length = int(t_max - t_min)
        beginning = t_min - (duration - length) / 2
        if beginning < 0:
            beginning = 0

        end = beginning + duration
        if end > signal.shape[0]:
            end = signal.shape[0]

        sound_slice = signal[int(beginning):int(end)]

        mel_spec = lb.feature.melspectrogram(sound_slice, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                             fmin=min_fr, fmax=max_fr, power=power)
        mel_spec = resize(mel_spec, new_shape)
        mel_spec = np.reshape(mel_spec, (mel_spec.shape[0], mel_spec.shape[1], 1))
        np.save(row['recording_id'] + '_' + row['species_id'], mel_spec)


def prep_test_spec_files(test_path, duration, stride, n_fft=2048, hop_length=512, new_shape=(128, 500), power=1.5):
    min_fr = 93.75 * 0.9
    max_fr = 13687.5 * 1.1
    root, dirnames, filenames = next(iter(os.walk(test_path)))
    for file in filenames:
        spec_batch = []
        signal, sr = lb.load(os.path.join(TEST_DATA_PATH, file), sr=None)
        start = 0
        end = int(duration * sr)
        while end <= signal.shape[0]:
            signal_bite = signal[start:end]
            new_spec = lb.feature.melspectrogram(signal_bite, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                 fmin=min_fr, fmax=max_fr, power=power)
            new_spec = resize(new_spec, new_shape)
            new_spec = np.reshape(new_spec, (new_spec.shape[0], new_spec.shape[1], 1))
            spec_batch.append(new_spec)

            start += int(stride * sr)
            end = int(start + duration * sr)
        file_name = file.split('.')[0]
        np.save(file_name, np.stack(spec_batch))



# 1 hour to process all data
# 12.44 GB for all data
