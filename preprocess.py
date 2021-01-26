"""Preprocess dataset"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import numpy as np
from tqdm import tqdm

from audio import Audio


def read_config(cfg_file):
    """Read the configuration from file and return it as a dictionary
    """
    with open(cfg_file, "r") as file_reader:
        cfg = json.load(file_reader)

    return cfg


def _process_utterance(audio_processor, qwav_dir, mel_dir, wav_path):
    """Preprocesses a single audio utterance. This will write the quantized wav and mel scale spectrogram to disk and
        returns a tuple to write to the train.txt file
        Args:
            audio_processor (object): The audio processor instance
            qwav_dir: The directory to write the quantized wav into
            mel_dir: The dicrectory to write the mel spectrogram into
            wav_path: Path to the audio file containing the speech input
        Returns:
            A (filename, num_frames) tuple to write to train.txt
    """
    filename = os.path.splitext(os.path.basename(wav_path))[0]

    wav = audio_processor.load_wav(wav_path)

    # Compute the quantized wav and mel spectrogram
    qwav = audio_processor.mu_compression(wav)
    mel = audio_processor.compute_melspectrogram(wav)

    # Write to to disk
    np.save(os.path.join(qwav_dir, filename + ".npy"),
            qwav,
            allow_pickle=False)

    np.save(os.path.join(mel_dir, filename + ".npy"), mel, allow_pickle=False)

    return (filename, mel.shape[-1])


def build_from_path_ljspeech(cfg,
                             in_dir,
                             out_dir,
                             num_workers=1,
                             tqdm=lambda x: x):
    """Preprocesses the LJ Speech dataset from a given input path into a given output directory.
      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar
      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    """
    qwav_dir = os.path.join(out_dir, "qwav")
    os.makedirs(qwav_dir, exist_ok=True)

    mel_dir = os.path.join(out_dir, "mel")
    os.makedirs(mel_dir, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    audio_processor = Audio(cfg["audio"]["sampling_rate"],
                            cfg["audio"]["n_fft"], cfg["audio"]["win_length"],
                            cfg["audio"]["hop_length"], cfg["audio"]["n_mels"],
                            cfg["audio"]["fmin"], cfg["audio"]["num_bits"],
                            cfg["audio"]["ref_db_level"],
                            cfg["audio"]["max_db_level"])

    with open(os.path.join(in_dir, "metadata.csv"), "r") as file_reader:
        for line in file_reader:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, "wavs", f"{parts[0]}.wav")
            futures.append(
                executor.submit(
                    partial(_process_utterance, audio_processor, qwav_dir,
                            mel_dir, wav_path)))

    return [future.result() for future in tqdm(futures)]


def preprocess(cfg, in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)

    if cfg["dataset"] == "ljspeech":
        metadata = build_from_path_ljspeech(cfg,
                                            in_dir,
                                            out_dir,
                                            num_workers,
                                            tqdm=tqdm)
    else:
        raise NotImplementedError

    write_metadata(cfg, metadata, out_dir)


def write_metadata(cfg, metadata, out_dir):
    with open(os.path.join(out_dir, "train.txt"), "w") as file_writer:
        for m in metadata:
            file_writer.write('|'.join([str(x) for x in m]) + '\n')

    frames = sum([m[1] for m in metadata])
    frame_shift_ms = cfg["audio"]["hop_length"] / cfg["audio"][
        "sampling_rate"] * 1000
    hours = frames * frame_shift_ms / (3600 * 1000)

    print(
        f"Wrote {len(metadata)} utterances, {frames} frames ({hours:2f} hours)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")

    parser.add_argument("--cfg_file",
                        help="Path to the configuration file",
                        required=True)

    parser.add_argument("--dataset_dir",
                        help="Path to the dataset dir",
                        required=True)

    parser.add_argument("--out_dir",
                        help="Path to the output dir",
                        required=True)

    args = parser.parse_args()
    num_workers = cpu_count()

    cfg = read_config(args.cfg_file)

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir

    preprocess(cfg, dataset_dir, out_dir, num_workers)
