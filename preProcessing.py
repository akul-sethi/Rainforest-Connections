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


def create_spectogram(s, n_fft, hop_length):
    stft = lb.core.stft(s, n_fft=n_fft, hop_length=hop_length)
    spectogram = np.abs(stft)
    db_spectogram = lb.amplitude_to_db(spectogram)
    return db_spectogram


def prepare_spec_dataset(sr, n_fft, hop_length):
    data = []

    for row in train_tp['recording_id']:
        signal, sr = lb.load(os.path.join(TRAIN_PATH, row) + '.flac', sr=sr)
        s = create_spectogram(signal, n_fft, hop_length)
        data.append(s)
        print("Samples converted: " + str(len(data)))

    with open('train_tp_specs.npy', 'wb') as f:
        np.save(f, np.stack(data, axis=0))


def prepare_labels():
    labels = []

    for species in train_tp['species_id']:
        labels.append(species)
        print("Labels converted: " + str(len(labels)))

    with open('train_tp_labels.npy', 'wb') as f:
        np.save(f, labels)


# recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max
# Minimum frequency is 93.75
# Maximum frequency is 13687.5
# Duration of largest sample is 7.923900000000003


def prep_mel_specs(sr, n_fft, hop_length):
    min_fr = 93.75 * 0.9
    max_fr = 13687.5 * 1.1
    sr = sr
    duration = 10 * sr
    for index, row in tqdm(train_tp.iterrows()):
        signal, sr = lb.load(os.path.join(TRAIN_PATH, row['recording_id']) + '.flac', sr=None)

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
                                             fmin=min_fr, fmax=max_fr, power=1.5)
        mel_spec = resize(mel_spec, (128, 500))
        mel_spec = np.reshape(mel_spec, (mel_spec.shape[0], mel_spec.shape[1], 1))
        np.save(row['recording_id'] + '_' + row['species_id'], mel_spec)


def prep_test_specs(sample_length, stride):
    n_fft = 2048
    hop_length = 512
    min_fr = 93.75 * 0.9
    max_fr = 13687.5 * 1.1
    root, dirnames, filenames = next(iter(os.walk(TEST_DATA_PATH)))
    for file in filenames:
        spec_batch = []
        signal, sr = lb.load(os.path.join(TEST_DATA_PATH, file), sr=None)
        start = 0
        end = int(sample_length * sr)
        while end <= signal.shape[0]:
            signal_bite = signal[start:end]
            new_spec = lb.feature.melspectrogram(signal_bite, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                 fmin=min_fr, fmax=max_fr, power=1.5)
            new_spec = resize(new_spec, (128, 500))
            new_spec = np.reshape(new_spec, (new_spec.shape[0], new_spec.shape[1], 1))
            spec_batch.append(new_spec)

            start += int(stride * sr)
            end = int(start + sample_length * sr)
        np.save(file, np.stack(spec_batch))


prep_test_specs(10,10)

# 1 hour to process all data
# 12.44 GB for all data
