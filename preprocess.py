"""Dataset preprocessing"""

import argparse
import os

import numpy as np

from audio import Audio
from config import Config as cfg


def process_data(out_dir, wav_dir):
    """Process wav data in wav_dir and write the output features to out_dir
    """
    # Create output dirs; to write the features
    mel_dir = os.path.join(out_dir, "mel")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Instantiate the audio processor
    ap = Audio(sampling_rate=cfg.sampling_rate,
               preemphasis_factor=cfg.preemphasis_factor,
               min_db_level=cfg.min_db_level,
               ref_db_level=cfg.ref_db_level,
               fft_size=cfg.fft_size,
               win_length=cfg.win_length,
               hop_size=cfg.hop_size,
               num_mels=cfg.num_mels,
               fmin=cfg.fmin,
               fmax=cfg.fmax)

    # Process each wav file in wav_dir
    wav_ids = []
    for wavfile in os.listdir(wav_dir):
        filename = os.path.splitext(wavfile)[0]

        print(f"Processing .... {filename} ....")
        wav_ids.append(filename)

        wav = ap.load_wav(os.path.join(wav_dir, filename + ".wav"))
        mel = ap.melspectrogram(wav)
        wav = wav.astype(np.float32)

        np.save(os.path.join(raw_dir, filename + ".npy"), wav)
        np.save(os.path.join(mel_dir, filename + ".npy"), mel)

    # Write the wav ids to disk
    file_writer = open(os.path.join(out_dir, "train.txt"), "w")
    for id in wav_ids:
        file_writer.write(id + "\n")
    file_writer.close()


if __name__ == "__main__":
    #  Setup a command line parser to get and parse command line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for training WaveRNN model")

    parser.add_argument(
        "--wav_dir",
        help="Path to the dir containing the dataset wav files",
        required=True)

    parser.add_argument(
        "--out_dir",
        help="Path to the output dir, where processed features will be written",
        required=True)

    args = parser.parse_args()

    wav_dir = args.wav_dir
    out_dir = args.out_dir

    # Process data
    process_data(out_dir, wav_dir)
