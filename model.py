"""WaveRNN model"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_GRUCell(gru_layer):
    """Instantiate GRUCell with the same paramters as the GRU layer
    """
    gru_cell = nn.GRUCell(gru_layer.input_size, gru_layer.hidden_size)

    gru_cell.weight_hh.data = gru_layer.weight_hh_l0.data
    gru_cell.weight_ih.data = gru_layer.weight_ih_l0.data
    gru_cell.bias_hh.data = gru_layer.bias_hh_l0.data
    gru_cell.bias_ih.data = gru_layer.bias_ih_l0.data

    return gru_cell


class WaveRNN(nn.Module):
    """WaveRNN model
    """
    def __init__(self, n_mels, hop_length, num_bits, audio_embedding_dim,
                 conditioning_rnn_size, rnn_size, fc_size):
        """Instantiate the WaveRNN model
        """
        super().__init__()

        self.n_mels = n_mels
        self.hop_length = hop_length
        self.num_bits = num_bits
        self.audio_embedding_dim = audio_embedding_dim
        self.conditioning_rnn_size = conditioning_rnn_size
        self.rnn_size = rnn_size
        self.fc_size = fc_size

        # Conditioning network
        self.conditioning_network = nn.GRU(input_size=n_mels,
                                           hidden_size=conditioning_rnn_size,
                                           num_layers=2,
                                           batch_first=True,
                                           bidirectional=True)

        # Quantized audio embedding
        self.quantized_audio_embedding = nn.Embedding(
            num_embeddings=2**num_bits, embedding_dim=audio_embedding_dim)

        # Autoregressive RNN
        self.rnn = nn.GRU(input_size=audio_embedding_dim +
                          2 * conditioning_rnn_size,
                          hidden_size=rnn_size,
                          batch_first=True)

        # Affine layers
        self.linear_layer = nn.Linear(in_features=rnn_size,
                                      out_features=fc_size)

        self.output_layer = nn.Linear(in_features=fc_size,
                                      out_features=2**num_bits)

    def forward(self, qwavs, mels):
        """Forward pass
        """
        # Conditioning network
        mels, _ = self.conditioning_network(mels)

        # Upsampling
        mels = F.interpolate(mels.transpose(1, 2),
                             scale_factor=self.hop_length)
        mels = mels.transpose(1, 2)

        # Quantized audio embedding
        embedded_qwavs = self.quantized_audio_embedding(qwavs)

        # Autoregressive RNN
        x, _ = self.rnn(torch.cat((embedded_qwavs, mels), dim=2))

        x = self.output_layer(F.relu(self.linear_layer(x)))

        return x

    def generate(self, mel):
        """Inference mode (Generates an audio waveform from a mel-spectrogram)
        """
        wav = []
        gru_cell = _init_GRUCell(self.rnn)

        # Conditioning network
        mel, _ = self.conditioning_network(mel)

        # Upsampling
        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        h = torch.zeros(mel.size(0), self.rnn_size, device=mel.device)
        x = torch.zeros(mel.size(0), device=mel.device, dtype=torch.long)
        x = x.fill_(2**(self.bits - 1))

        for mel_frame in torch.unbind(mel, dim=1):
            # Audio embedding
            x = self.quantized_audio_embedding(x)

            # Autoregressive GRU Cell
            h = gru_cell(torch.cat((x, mel_frame), dim=1), h)

            x = F.relu(self.linear_layer(x))
            logits = self.output_layer(x)

            # Apply softmax over the logits and generate a distribution
            posterior = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(posterior)

            # Sample from the distribution to generate output
            x = dist.sample()
            wav.append(x.item())

        wav = np.asarray(wav, dtype=np.int)
        wav = librosa.mu_expand(wav - 2**(self.num_bits - 1),
                                mu=2**self.num_bits - 1)

        return wav
