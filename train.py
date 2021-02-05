"""WaveRNN training"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import WaveRNNDataset
from model import WaveRNN


def read_cfg_file(cfg_file):
    """Read the configuration from file and return it as a dictionary
    """
    with open(cfg_file, "r") as file_reader:
        cfg = json.load(file_reader)

    return cfg


def save_checkpoint(checkpoint_dir, model, optimizer, scheduler, step):
    """Write a training checkpoint to disk
    """
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }

    checkpoint_path = os.path.join(checkpoint_dir,
                                   f"checkpoint-step{step:09d}.pth")

    torch.save(checkpoint_state, checkpoint_path)
    print(f"{checkpoint_path} written to disk")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load a checkpoint from disk
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict[checkpoint["scheduler"]]

    return checkpoint["step"]


def train_model(cfg, data_dir, checkpoints_dir, logs_dir, chcekpoint_path):
    """Train the WaveRNN model
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Instantiate the dataset
    dataset = WaveRNNDataset(data_root_dir=data_dir,
                             hop_length=cfg["audio"]["hop_length"],
                             sample_frames=cfg["training"]["sample_frames"])

    # Setup the dataloader
    data_loader = DataLoader(dataset,
                             batch_size=cfg["training"]["batch_size"],
                             shuffle=True,
                             num_workers=cfg["training"]["num_workers"],
                             pin_memory=True,
                             drop_last=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup WaveRNN model
    model = WaveRNN(
        n_mels=cfg["audio"]["n_mels"],
        conditioning_rnn_size=cfg["model"]["conditioning_rnn_size"],
        audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
        rnn_size=cfg["model"]["rnn_size"],
        fc_size=cfg["model"]["fc_size"],
        num_bits=cfg["audio"]["num_bits"],
        hop_length=cfg["audio"]["hop_length"])

    # Place the model on the appropriate device
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["training"]["learning_rate"])

    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, cfg["training"]["lr_scheduler"]["step_size"],
        cfg["training"]["lr_scheduler"]["gamma"])

    # Initialize the step
    if checkpoint_path:
        global_step = load_checkpoint(checkpoint_path, model, optimizer,
                                      lr_scheduler)
    else:
        global_step = 0

    num_epochs = cfg["training"]["num_steps"] // len(data_loader) + 1

    # Initialize the starting epoch
    start_epoch = global_step // len(data_loader) + 1

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = 0.0
        for idx, (mel, quantized_audio) in enumerate(data_loader, 1):

            quantized_audio, mel = quantized_audio.to(device), mel.to(device)

            optimizer.zero_grad()

            # Compute the loss
            audio = model(quantized_audio[:, :-1], mel)
            loss = F.cross_entropy(audio.transpose(1, 2), quantized_audio[:,
                                                                          1:])

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if global_step % cfg["training"]["checkpoint_interval"] == 0:
                save_checkpoint(checkpoints_dir, model, optimizer,
                                lr_scheduler, global_step)

            avg_loss += (loss.item() - avg_loss) / idx

            # Increment the step
            global_step += 1

        # Log the training progress
        print(
            f"Epoch: {epoch:09d}, Step: {global_step:09d}, Loss: {avg_loss:03f}, {lr_scheduler.get_last_lr()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WaveRNN model")

    parser.add_argument(
        "--cfg_file",
        help="Path to the file containing the configuration / hyperparameters",
        required=True)

    parser.add_argument(
        "--data_dir",
        help="Path to the data dir containing the training data",
        required=True)

    parser.add_argument(
        "--checkpoints_dir",
        help="Path to the dir where the training checkpoints will be saved",
        required=True)

    parser.add_argument(
        "--logs_dir",
        help="Path to the dir where the training logs will be written",
        required=True)

    parser.add_argument(
        "--checkpoint_path",
        help="If specified load checkpoint and restart training from that point"
    )

    args = parser.parse_args()

    cfg = read_cfg_file(args.cfg_file)
    data_dir = args.data_dir
    checkpoints_dir = args.checkpoints_dir
    logs_dir = args.logs_dir
    checkpoint_path = args.checkpoint_path

    print("Training started")

    # Train the model
    train_model(cfg, data_dir, checkpoints_dir, logs_dir, checkpoint_path)

    print("Training Complete")
    sys.exit(0)
