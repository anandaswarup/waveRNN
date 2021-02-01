"""WaveRNN model"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_gru_cell(gru_layer):
    """Initialize GRUCell
    """
    gru_cell = nn.GRUCell(gru_layer.input_size, gru_layer.hidden_size)

    gru_cell.weight_hh.data = gru_layer.weight_hh_l0.data
    gru_cell.weight_ih.data = gru_layer.weight_ih_l0.data
    gru_cell.bias_hh.data = gru_layer.bias_hh_l0.data
    gru_cell.bias_ih.data = gru_layer.bias_ih_l0.data

    return gru_cell


class WaveRNN(nn.Module):
    """WaveRNN vocoder model
    """
    def __init__(self, n_mels, conditioning_rnn_size, audio_embedding_dim,
                 rnn_size, fc_size, num_bits, hop_length):
        """Initialize the vocoder model
        """
        super().__init__()

        self.n_mels = n_mels
        self.conditioning_rnn_size = conditioning_rnn_size
        self.audio_embedding_dim = audio_embedding_dim
        self.rnn_size = rnn_size
        self.fc_size = fc_size
        self.num_bits = num_bits
        self.hop_length = hop_length

        # Conditioning network
        self.conditioning_network = nn.GRU(n_mels,
                                           conditioning_rnn_size,
                                           num_layers=2,
                                           batch_first=True,
                                           bidirectional=True)

        # Autoregressive network
        self.audio_embedding = nn.Embedding(num_embeddings=2**num_bits,
                                            embedding_dim=audio_embedding_dim)
        self.rnn = nn.GRU(audio_embedding_dim + 2 * conditioning_rnn_size,
                          rnn_size,
                          batch_first=True)
        self.fc1 = nn.Linear(rnn_size, fc_size)
        self.fc2 = nn.Linear(fc_size, 2**self.num_bits)

    def forward(self, quantized_audio, mel):
        """Forward pass for training
            Args:
                mel (tensor): Mel spectrogram of shape [batch_size, seq_len, n_mels]
                quantized_audio (tensor): Mu-law quantized audio [batch_size, seq_len, 2 ** num_bits - 1] 
        """
        # Conditioning network
        mel, _ = self.conditioning_network(mel)

        # Upsampling
        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        # Autoregressive network
        x = self.audio_embedding(quantized_audio)
        x, _ = self.rnn(torch.cat((x, mel), dim=2))
        x = self.fc2(F.relu(self.fc1(x)))

        return x

    def generate(self, mel):
        """Generate an audio waveform from a mel-spectrogram
            Args:
                mel (tensor): Mel spectrogram of shape [1, seq_len, n_mels]
            Returns:
                (numpy array): Generated waveform of shape [seq_len * hop_length]
        """
        wav_hat = []

        rnn_cell = _init_gru_cell(self.rnn)

        # Conditioning network
        mel, _ = self.conditioning_network(mel)

        # Upsampling
        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        # Autoregressive network
        hidden = torch.zeros(mel.size(0), self.rnn_size, device=mel.device)

        x = torch.zeros(mel.size(0), device=mel.device, dtype=torch.long)
        x = x.fill_(2**(self.num_bits - 1))

        for frame in torch.unbind(mel, dim=1):
            x = self.audio_embedding(x)
            hidden = rnn_cell(torch.cat((x, frame), dim=1), hidden)

            logits = self.fc2(F.relu(self.fc1(x)))
            posterior = F.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(posterior)

            x = distribution.sample()
            wav_hat.append(x.item())

        wav_hat = np.asarray(wav_hat, dtype=np.int)
        wav_hat = librosa.mu_expand(wav_hat - 2**(self.num_bits - 1),
                                    mu=(2**self.num_bits - 1))

        return wav_hat
