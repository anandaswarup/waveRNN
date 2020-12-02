"""Audio processor"""

import math

import librosa
import numpy as np
import scipy


class Audio():
    """Audio processor class
    """
    def __init__(self, sampling_rate, n_fft, win_length, hop_length, num_mels,
                 fmin, fmax):
        """Initialize the audio processor
        """
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.min_db_level = -100
        self.ref_db_level = 20

    def encode(self, x):
        """Encode x using 16 bits
        """
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    def load_wav(self, wavpath, encode=True):
        """Load wav file from disk
        """
        y = librosa.load(wavpath, sr=self.sampling_rate)[0]
        if encode is True:
            y = self.encode(y)

        return y

    def save_wav(self, y, wavpath):
        """Write wav file to disk
        """
        if y.dtype != "int16":
            y = self.encode(y)

        scipy.io.wavfile.write(wavpath, self.sampling_rate, y.astype(np.int16))
        print(f"Written {wavpath} to disk")

    def split_signal(y):
        """Split the signal to coarse and file components
        """
        unsigned = y + 2**15

        coarse = unsigned // 256
        fine = unsigned % 256

        return coarse, fine

    def combine_signal(self, coarse, fine):
        """Combine the coarse and fine components to get the signal
        """
        return coarse * 256 + fine - 2**15

    def stft(self, y):
        """Compute the STFT of signal
        """
        return librosa.stft(y=y,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length)

    def amp_to_db(x):
        """Transform amplitudes to decibels
        """
        return 20 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(x):
        """Transform decibels to amplitudes
        """
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        """Normalize values of S to [0, 1] range
        """
        return np.clip((S - self.min_db_level) / -self.min_db_level, 0, 1)

    def denormalize(self, S):
        """Denormalize values from [0, 1] range back to original values range
        """
        return (np.clip(S, 0, 1) * -self.min_db_level) + self.min_db_level

    def linear_to_mel(self, S):
        """Transform linear scale spectrogram to mel scale spectrogram
        """
        mel_basis = librosa.filters.mel(self.sampling_rate,
                                        n_fft=self.n_fft,
                                        n_mels=self.num_mels,
                                        fmin=self.fmin,
                                        fmax=self.fmax)

        return np.dot(mel_basis, S)

    def spectrogram(self, y):
        """Compute linear scale log magnitude spectrogram from signal
        """
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.ref_db_level

        return self.normalize(S)

    def melspectrogram(self, y):
        """Compute mel scale log magnitude spectrogram from signal
        """
        D = self.stft(y)
        S = self.amp_to_db(self.linear_to_mel(np.abs(D)))

        return self.normalize(S)
