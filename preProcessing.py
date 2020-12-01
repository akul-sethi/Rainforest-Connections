# packages
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa as lb
import json
import librosa.display
import matplotlib.pyplot as plt
import os

TRAIN_PATH = "rfcx-species-audio-detection/train/"


def create_spectogram(s, n_fft, hop_length):
    stft = lb.core.stft(s, n_fft=n_fft, hop_length=hop_length)
    spectogram = np.abs(stft)
    db_spectogram = lb.amplitude_to_db(spectogram)
    return db_spectogram


def prepare_spec_dataset(sr, n_fft, hop_length):
    train_tp = pd.read_csv("rfcx-species-audio-detection/train_tp.csv", delimiter=',')
    train_fp = pd.read_csv("rfcx-species-audio-detection/train_fp.csv", delimiter=',')

    data = {
        "Inputs": [],
        "Labels": []
    }

    for i, row in enumerate(train_tp.itertuples()):
        signal, sr = lb.load(os.path.join(TRAIN_PATH, row.recording_id) + '.flac', sr=sr)
        s = create_spectogram(signal, n_fft, hop_length)
        data["Inputs"].append(s.tolist())

    for i, row in enumerate(train_tp.itertuples()):
        v = [0 for i in range(24)]
        v[row.species_id] = 1
        data["Labels"].append(v)

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)


# recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max

prepare_spec_dataset(22050, 2048, 512)
