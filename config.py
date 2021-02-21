"""Configuration settings"""


class Config:
    dataset = "ljspeech"

    # Audio processing parameters
    sampling_rate = 22050
    max_db = 100
    ref_db = 20

    n_fft = 2048
    win_length = 1100  # 50 ms window length
    hop_length = 275  # 12.5 ms frame shift

    num_mels = 80
    fmin = 50

    num_bits = 10  # Bit depth of the signal

    # Model parameters
    conditioning_rnn_size = 128
    audio_embedding_dim = 256
    rnn_size = 896
    fc_size = 1024

    # Training
    batch_size = 16
    num_steps = 200000
    sample_frames = 24
    learning_rate = 4e-4
    lr_scheduler_step_size = 25000
    lr_scheduler_gamma = 0.5
    checkpoint_interval = 25000
    num_workers = 8
