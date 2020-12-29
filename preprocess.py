"""Data preprocessing"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import numpy as np
import yaml
from tqdm import tqdm

from audio import AudioProcessor


def load_config(filename):
    """Load configuration from yaml file and return it as a dictionary
    """
    with open(filename, "r") as file_reader:
        cfg = yaml.load(file_reader, Loader=yaml.FullLoader)

    return cfg


def _process_utterance(audio_processor, mel_dir, quantized_wav_dir, wav_path):
    """Process a single wav file
    This writes the mel spectrogram and quantized wav file to disk and returns a tuple to write to the the train.txt
    file
        Args:
            audio_processor: The audio processor
            mel_dir: The directory to write the mel spectrogram to
            quantized_wav_dir: The directory to write the quantized wav file to
            wav_path: Path to to the wav file to be processed

        Returns:
            A (filename, num_frames) tuple to write to train.txt
    """
    filename = os.path.splitext(os.path.basename(wav_path))[0]

    # Load the wavfile to a numpy array
    wav = audio_processor.load_wav(wav_path)

    # Compute mel-spectrogram from wav
    mel = audio_processor.mel_spectrogram(wav).astype(np.float32)

    # Compute a quantized representation of wav
    quantized_wav = audio_processor.mulaw_compression(wav)

    # Write to disk
    np.save(os.path.join(mel_dir, filename + ".npy"), mel, allow_pickle=False)

    np.save(os.path.join(quantized_wav_dir, filename + ".npy"),
            quantized_wav,
            allow_pickle=False)

    return filename, mel.shape[-1]


def build_path_from_ljspeech(cfg,
                             in_dir,
                             out_dir,
                             num_workers=1,
                             tqdm=lambda x: x):
    """Preprocess the LJ speech dataset from a given input path into a given output directory
        Args:
            in_dir: The directory where the LJSpeech dataset has been downloaded
            out_dir: The directory to write the output to
            num_workers: Optional number of workers processes to parallelize across
            tqdm: Optional argument to get a nice progress bar
    """
    mel_dir = os.path.join(out_dir, "mel")
    os.makedirs(mel_dir, exist_ok=True)

    quantized_wav_dir = os.path.join(out_dir, "qwav")
    os.makedirs(quantized_wav_dir, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    # Instantiate the audio processor
    audio_processor = AudioProcessor(
        sampling_rate=cfg["audio"]["sampling_rate"],
        preemph_factor=cfg["audio"]["preemph_factor"],
        max_db_level=cfg["audio"]["max_db_level"],
        ref_db_level=cfg["audio"]["ref_db_level"],
        n_fft=cfg["audio"]["n_fft"],
        win_length=cfg["audio"]["win_length"],
        hop_length=cfg["audio"]["hop_length"],
        n_mels=cfg["audio"]["n_mels"],
        num_bits=cfg["audio"]["num_bits"])

    with open(os.path.join(in_dir, "metadata.csv"), "r") as file_reader:
        for line in file_reader:
            parts = line.strip().split("|")
            wav_path = os.path.join(in_dir, "wavs", f"{parts[0]}.wav")
            futures.append(
                executor.submit(
                    partial(_process_utterance, audio_processor, mel_dir,
                            quantized_wav_dir, wav_path)))

    return [future.result() for future in tqdm(futures)]


def preprocess(cfg, in_dir, out_dir, num_workers):
    """Preprocess the dataset
    """
    os.makedirs(out_dir, exist_ok=True)

    if cfg["dataset"] == "ljspeech":
        metadata = build_path_from_ljspeech(cfg,
                                            in_dir,
                                            out_dir,
                                            num_workers,
                                            tqdm=tqdm)
    else:
        raise NotImplementedError

    write_metadata(cfg, metadata, out_dir)


def write_metadata(cfg, metadata, out_dir):
    """Write the metadata to the train.txt file in the output dir
    """
    with open(os.path.join(out_dir, "train.txt"), "w") as file_writer:
        for m in metadata:
            file_writer.write(str(m[0]) + "\n")

    frames = sum([m[1] for m in metadata])
    frame_shift_ms = cfg["audio"]["hop_length"] / cfg["audio"][
        "sampling_rate"] * 1000
    hours = frames * frame_shift_ms / (3600 * 1000)

    print(
        f"Wrote {len(metadata)} utterances, {frames} frames ({hours:2f} hours)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")

    parser.add_argument("--config_file",
                        help="Path to the configuration file (in yaml format)",
                        required=True)

    parser.add_argument("--dataset_dir",
                        help="Path to the dataset dir",
                        required=True)

    parser.add_argument("--out_dir",
                        help="Path to the output dir",
                        required=True)

    args = parser.parse_args()
    num_workers = cpu_count()

    cfg = load_config(args.config_file)
    dataset_dir = args.dataset_dir
    out_dir = args.out_dir

    preprocess(cfg, dataset_dir, out_dir, num_workers)
