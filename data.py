"""WaveRNN Dataset"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def load_training_instances(filename, mel_dir, qwav_dir):
    """Load training instances from train.txt file
    """
    with open(filename, "r") as file_reader:
        file_data = file_reader.readlines()

    training_ids = [line.split("|")[0] for line in file_data]
    training_instances = [(os.path.join(mel_dir, id + ".npy"),
                           os.path.join(qwav_dir, id + ".npy"))
                          for id in training_ids]

    return training_instances


class WaveRNNDataset(Dataset):
    """WaveRNN Dataset
    """
    def __init__(self, data_root_dir, hop_length, sample_frames):
        """Initialize the WaveRNN Dataset
        """
        self.hop_length = hop_length
        self.sample_frames = sample_frames
        self.training_instances = load_training_instances(
            os.path.join(data_root_dir, "train.txt"),
            os.path.join(data_root_dir, "mel"),
            os.path.join(data_root_dir, "qwav"))

    def __len__(self):
        return len(self.training_instances)

    def __getitem__(self, index):
        mel_path, qwav_path = self.training_instances[index]

        # Load the mel spectrogram and quantized wav file from disk
        qwav = np.load(qwav_path)
        mel = np.load(mel_path)

        pos = random.randint(0, mel.shape[-1] - self.sample_frames - 1)

        mel = mel[:, pos:pos + self.sample_frames]

        p, q = pos, pos + self.sample_frames
        qwav = qwav[p * self.hop_length:q * self.hop_length + 1]

        return torch.FloatTensor(mel.T), torch.LongTensor(qwav)
