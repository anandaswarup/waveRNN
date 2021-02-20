"""WaveRNN dataset"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config as cfg


def _load_training_instances(filename):
    """Load the training instances from disk
    """
    with open(filename, "r") as file_reader:
        training_instances = file_reader.readlines()

    training_instances = [
        instance.strip("\n") for instance in training_instances
    ]

    training_instances = [
        instance.split("|") for instance in training_instances
    ]

    return training_instances


class VocoderDataset(Dataset):
    """Vocoder dataset
    """
    def _init__(self, data_dir):
        """Instantiate the dataset
        """
        self.training_instances = _load_training_instances(
            os.path.join(data_dir, "train.txt"))

        self.sample_frames = cfg.sample_frames
        self.hop_length = cfg.hop_length

    def __len__(self):
        return len(self.training_instances)

    def __getitem__(self, index):
        mel_path, qwav_path, _ = self.training_instances[index][
            0], self.training_instances[index][1], self.training_instances[
                index][2]

        mel = np.load(mel_path)
        qwav = np.load(qwav_path)

        pos = random.randint(0, mel.shape[-1] - self.sample_frames - 1)
        mel = mel[:, pos:pos + self.sample_frames]

        p, q = pos, pos + self.sample_frames
        qwav = qwav[p * self.hop_length:q * self.hop_length + 1]

        return torch.FloatTensor(mel.T), torch.LongTensor(qwav)
