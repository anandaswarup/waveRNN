"""Audio processing utilities"""

import librosa
import numpy as np


class AudioProcessor():
    """Audio processing class
    """
    def __init__(self, sampling_rate, preemph_factor, max_db_level,
                 ref_db_level, n_fft, win_length, hop_length, n_mels,
                 num_bits):
        """Initialize the audio processor
        """
        self.sampling_rate = sampling_rate
        self.preemph_factor = preemph_factor
        self.max_db_level = max_db_level
        self.ref_db_level = ref_db_level
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_bits = num_bits

    def _apply_preemphasis(self, wav):
        """Apply pre-emphasis to the wav signal
        """
        return librosa.effects.preemphasis(wav, coef=self.preemph_factor)

    def load_wav(self, wavpath):
        """Load the wav file from disk into memory
        """
        wav, _ = librosa.load(wavpath, sr=self.sampling_rate)
        peak = np.abs(wav).max()
        if peak >= 1:
            wav = wav / peak * 0.999

        return wav

    def mel_spectrogram(self, wav):
        """Compute log magnitude mel spectrogram from wav signal
        """
        # Apply pre-emphasis to the signal
        wav = self._apply_preemphasis(wav)

        # Compute the mel scale spectrogram from signal
        mel = librosa.feature.melspectrogram(wav,
                                             sr=self.sampling_rate,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length,
                                             n_fft=self.n_fft,
                                             n_mels=self.n_mels,
                                             norm=1,
                                             power=1)

        mel = librosa.amplitude_to_db(mel, top_db=None) - self.ref_db_level
        mel = np.maximum(mel, -self.max_db_level)

        return mel / self.max_db_level

    def mulaw_compression(self, wav):
        """Apply mu-law compression on the signal
        """
        wav = np.pad(wav, (self.win_length // 2), mode="reflect")
        wav = wav[:((wav.shape[0] - self.win_length) // self.hop_length + 1) *
                  self.hop_length]
        wav = 2**(self.num_bits - 1) + librosa.mu_compress(
            wav, mu=2**self.num_bits - 1)

        return wav
