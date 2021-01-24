"""Audio processing utilities"""

import librosa
import numpy as np


class Audio():
    """Audio processing class
    """
    def __init__(self, sampling_rate, n_fft, win_length, hop_length, n_mels,
                 fmin, num_bits, ref_db_level, max_db_level):
        """Initialize the Audio Processor
        """
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.num_bits = num_bits
        self.ref_db_level = ref_db_level
        self.max_db_level = max_db_level

    def mu_compression(self, wav):
        """Compress the signal using mu-law compression
        """
        wav = np.pad(wav, (self.win_length // 2, ), mode="reflect")
        wav = wav[:((wav.shape[0] - self.win_length) // self.hop_length + 1) *
                  self.hop_length]
        wav = 2**(self.num_bits - 1) + librosa.mu_compress(
            wav, mu=(2**self.num_bits - 1))

        return wav

    def compute_melspectrogram(self, wav):
        """Compute log mel spectrogram from signal
        """
        # Apply pre-emphasis on wav
        wav = librosa.effects.preemphasis(wav, coef=0.97)

        # Compute mel spectrogram from wav
        mel = librosa.feature.melspectrogram(wav,
                                             sr=self.sampling_rate,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length,
                                             n_fft=self.n_fft,
                                             n_mels=self.n_mels,
                                             fmin=self.fmin,
                                             norm=1,
                                             power=1)

        mel = librosa.amplitude_to_db(mel, top_db=None) - self.ref_db_level
        mel = np.maximum(mel, -self.max_db_level)

        return mel / self.ref_db_level

    def load_wav(self, path):
        """Load the wav file from disk
        """
        wav, _ = librosa.load(path, sr=self.sampling_rate)

        peak = np.abs(wav).max()
        if peak > 1:
            wav = wav / peak * 0.999

        return wav
