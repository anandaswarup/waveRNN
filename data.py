"""Data loading and handling"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def load_training_items_from_file(filename):
    """Load the training items from train.txt file
    """
    with open(filename, "r") as file_reader:
        items = file_reader.readlines()
    items = [line.strip("\n") for line in items]

    return items


class VocoderDataset(Dataset):
    """Vocoder dataset
    """
    def __init__(self, data_root_dir, sample_frames=24, hop_length=275):
        """Initialize the dataset
        """
        self.root_dir = data_root_dir
        self.sample_frames = sample_frames
        self.hop_length = hop_length
        self.items = self.load_training_items_from_file(
            os.path.join(data_root_dir, "train.txt"))

    def __len__(self):
        """Return the length of the dataset
        """
        return len(self.items)

    def __getitem__(self, index):
        """Retrieve a particular element of the dataset (based on index)
        """
        fileid = self.items[index]

        wav = np.load(os.path.join(self.root_dir, "qwav", fileid + ".npy"))
        mel = np.load(os.path.join(self.root_dir, "mel", fileid + ".npy"))

        pos = random.randint(0, mel.shape[-1] - self.sample_frames - 1)
        mel = mel[:, pos:pos + self.sample_frames]

        p, q = pos, pos + self.sample_frames
        wav = wav[p * self.hop_length:q * self.hop_length + 1]

        return torch.LongTensor(wav), torch.LongTensor(mel.T)
