"""Model dataloader"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config as cfg


def _load_training_instances(filename):
    """Load training instances from train.txt file
    """
    with open(filename, "r") as file_reader:
        training_instances = file_reader.readlines()

    training_instances = [
        instance.strip("\n") for instance in training_instances
    ]

    return training_instances


class VocoderDataset(Dataset):
    """Vocoder dataset
    """
    def __init__(self, data_dir):
        """Instantiate the Vocoder Dataset
        """
        self.training_instances = _load_training_instances(
            os.path.join(data_dir, "train.txt"))

        self.raw_dir = os.path.join(data_dir, "raw")
        self.mel_dir = os.path.join(data_dir, "mel")

    def __len__(self):
        return len(self.training_instances)

    def __getitem__(self, index):
        train_id = self.training_instances[index]

        mel = np.load(os.path.join(self.mel_dir, train_id + ".npy"))
        raw = np.load(os.path.join(self.raw_dir, train_id + ".npy"))

        return mel, raw


def collate(batch):
    """Collate function used for batching
    """
    pad = 2
    mel_window = cfg.seq_len // cfg.hop_size + 2 * pad

    max_offsets = [x[0].shape[-1] - (mel_window + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    raw_offsets = [(offset + pad) * cfg.hop_size for offset in mel_offsets]

    mels = [
        x[0][:, mel_offsets[idx]:mel_offsets[idx] + mel_window]
        for idx, x in enumerate(batch)
    ]

    coarse = [
        x[1][raw_offsets[idx]:raw_offsets[idx] + cfg.seq_len + 1]
        for idx, x in enumerate(batch)
    ]

    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack * (coarse).astype(np.float32)

    mels = torch.FloatTensor(mels)
    coarse = torch.FloatTensor(coarse)

    x = coarse[:, :cfg.seq_len]
    y = coarse[:, 1:]

    return x, mels, y
