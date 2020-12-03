# packages
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa as lb
import json
import librosa.display
import matplotlib.pyplot as plt
import os

TRAIN_PATH = "rfcx-species-audio-detection/train/"

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

# prepare_spec_dataset(22050, 2048, 512)
# print("Done with Spectograms")
prepare_labels()
print("Done with everything")

# 1 hour to process all data
# 4.44 GB for all data
