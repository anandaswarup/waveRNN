"""WaveRNN generation"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch

from config import Config as cfg
from model import WaveRNN


def generate(checkpoint_path, eval_data_dir, out_dir):
    """Generate waveforms from mel-spectrograms using WaveRNN
    """
    os.makedirs(out_dir, exist_ok=True)

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = WaveRNN(n_mels=cfg.num_mels,
                    hop_length=cfg.hop_length,
                    num_bits=cfg.num_bits,
                    audio_embedding_dim=cfg.audio_embedding_dim,
                    conditioning_rnn_size=cfg.conditioning_rnn_size,
                    rnn_size=cfg.rnn_size,
                    fc_size=cfg.fc_size)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model_step = checkpoint["step"]

    for filename in open(os.path.join(eval_data_dir, "eval.txt"), "r"):
        filename = filename.strip("\n")
        print("Generating {filename}")

        mel = np.load(os.path.join(eval_data_dir, "mel", filename + ".npy"))
        mel = torch.FLoatTensor(mel.T).unsqueeze(0).to(device)

        with torch.no_grad():
            wav_hat = model.generate(mel)

        out_path = os.path.join(out_dir,
                                f"model_step{model_step:09d}_{filename}.wav")

        sf.write(out_path, wav_hat, cfg.sampling_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate waveforms from mel-spectrograms using WaveRNN")

    parser.add_argument(
        "--checkpoint_path",
        help="Path to the checkpoint to use to instantiate the model",
        required=True)

    parser.add_argument(
        "--eval_data_dir",
        help="Path to the dir containing the spectrograms to be synthesized",
        required=True)

    parser.add_argument(
        "--out_dir",
        help="Path to the dir where generated waveforms will be saved",
        required=True)

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    eval_data_dir = args.eval_data_dir
    out_dir = args.out_dir

    generate(checkpoint_path, eval_data_dir, out_dir)
