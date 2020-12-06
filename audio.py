"""Audio processing"""

import librosa
import lws
import numpy as np
from scipy import signal
from scipy.io import wavfile


class Audio():
    """Audio processing class
    """
    def __init__(self, sampling_rate, premephasis_factor, min_db_level,
                 ref_db_level, fft_size, win_length, hop_size, num_mels, fmin,
                 fmax):
        """Initalize the audio processing class
        """
        self.sampling_rate = sampling_rate
        self.premephasis_factor = premephasis_factor
        self.min_db_level = min_db_level
        self.ref_db_level = ref_db_level
        self.fft_size = fft_size
        self.win_length = win_length
        self.hop_size = hop_size
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax

    def _lws_processor(self):
        """Get the lws processor object to compute the STFT
        """
        return lws.lws(self.fft_size, self.hop_size, mode="speech")

    def _linear_to_mel(self, S):
        """Linear to mel conversion
        """
        if self.fmax is not None:
            assert self.fmax <= self.sampling_rate // 2

        mel_basis = librosa.filters.mel(self.sampling_rate,
                                        self.fft_size,
                                        fmin=self.fmin,
                                        fmax=self.fmax,
                                        n_mels=self.num_mels)

        return np.dot(mel_basis, S)

    def _amp_to_db(self, x):
        """Convert amplitudes to decibels
        """
        min_level = np.exp(self.min_db_level / 20 * np.log(10))

        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        """Convert decibels to amplitudes
        """
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        """Normalize spectrogram values to lie between [0, 1]
        """
        return np.clip((S - self.min_db_level) / -self.min_db_level, 0, 1)

    def _denormalize(self, S):
        """Recover original spectrogram values back from [0, 1] range
        """
        return (np.clip(S, 0, 1) * -self.min_db_level) + self.min_db_level

    def load_wav(self, wavpath):
        """Load wavfile from disk into memory
        """
        return librosa.load(wavpath, sr=self.sampling_rate)[0]

    def save_wav(self, y, wavpath):
        """Save signal to disk as wavfile
        """
        y = y * 32767 / max(0.01, np.max(np.abs(y)))
        wavfile.write(wavpath, self.sampling_rate, y.astype(np.int16))

    def preemphasis(self, y):
        """Apply preemphasis to the signal
        """
        b = np.array([1., -self.premephasis_factor], y.dtype)
        a = np.array([1.], y.dtype)

        return signal.lfilter(b, a, y)

    def inv_preemphasis(self, y):
        """Invert the pre-emphasis applied to the signal
        """
        b = np.array([1.], y.dtype)
        a = np.array([1., -self.premephasis_factor], y.dtype)

        return signal.lfilter(b, a, y)

    def melspectrogram(self, y):
        """Compute log-magnitude mel spectrogram from signal
        """
        D = self._lws_processor().stft(self.preemphasis(y)).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_db_level

        return self._normalize(S)
