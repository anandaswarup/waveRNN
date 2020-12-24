"""WaveRNN vocoder model"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gru_cell(gru_layer):
    """Returns a GRU Cell initialized with the same parameters as the GRU Layer (will be used in inference)
    """
    gru_cell = nn.GRUCell(gru_layer.input_size, gru_layer.hidden_size)

    gru_cell.weight_hh.data = gru_layer.weight_hh_l0.data
    gru_cell.weight_ih.data = gru_layer.weight.ih.l0.data

    gru_cell.bias_hh.data = gru_layer.bias_hh_l0.data
    gru_cell.bias_ih.data = gru_layer.bias_ih.l0.data

    return gru_cell


class WaveRNN(nn.Module):
    """WaveRNN vocoder model which predicts a waveform from a mel spectrogram
    """
    def __init__(self, hop_length, num_bits, audio_embedding_dim, rnn_size,
                 affine_size, mel_dim, conditioning_rnn_size):
        """Initialize the model
        """
        super().__init__()

        self.hop_length = hop_length
        self.num_bits = num_bits
        self.audio_embedding_dim = audio_embedding_dim
        self.rnn_size = rnn_size
        self.affine_size = affine_size
        self.mel_dim = mel_dim
        self.conditioning_rnn_size = conditioning_rnn_size

        self.conditioning_rnn = nn.GRU(input_size=mel_dim,
                                       hidden_size=conditioning_rnn_size,
                                       num_layers=2,
                                       batch_first=True,
                                       bidirectional=True)

        self.audio_embedding = nn.Embedding(num_embeddings=2**num_bits,
                                            embedding_dim=audio_embedding_dim)

        self.rnn = nn.GRU(input_size=audio_embedding_dim +
                          2 * conditioning_rnn_size,
                          hidden_size=rnn_size,
                          batch_first=True)

        self.fc1 = nn.Linear(in_features=rnn_size, out_features=affine_size)
        self.fc2 = nn.Linear(in_features=affine_size, out_features=2**num_bits)

    def forward(self, quantized_audio, mels):
        """Forward pass
        """
        # Conditioning network
        mels, _ = self.conditioning_rnn(mels)

        # Upsampling the output of the conditioning network
        mels = F.interpolate(mels.transpose(1, 2),
                             scale_factor=self.hop_length)
        mels = mels.transpose(1, 2)

        # Autoregressive network
        x = self.audio_embedding(quantized_audio)
        x, _ = self.rnn(torch.cat((x, mels), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def inference(self, mel):
        """Inference mode
        """
        y_hat = []

        # Conditioning network
        mel, _ = self.conditioning_rnn(mel)

        # Upsampling the output of the conditioning network
        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        # Autoregressive network
        cell = get_gru_cell(self.rnn)
        h = torch.zeros(mel.size(0), self.rnn_size, device=mel.device)
        x = torch.zeros(mel.size(0), device=mel.device, dtype=torch.long)
        x = x.fill_(2**(self.num_bits - 1))

        for mel_frame in torch.unbind(mel, dim=1):
            x = self.embedding(x)
            h = cell(torch.cat((x, mel_frame), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)

            posterior = F.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(posterior)

            x = distribution.sample()
            y_hat.append(x.item())

        y_hat = np.asarray(y_hat, dtype=np.int)
        y_hat = librosa.mu_expand(y_hat - 2**(self.num_bits - 1),
                                  mu=2**self.num_bits - 1)

        return y_hat
