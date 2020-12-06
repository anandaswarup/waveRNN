"""Configuration"""


class Config:
    """waveRNN configuration class"""

    # Audio processing parameters
    sampling_rate = 22050
    preemphasis_factor = 0.97
    min_db_level = -100
    ref_db_level = 20

    fft_size = 1024
    win_length = 1024
    hop_size = 256

    num_mels = 80
    fmin = 125
    fmax = 7600

    rescaling = False
    rescaling_max = 0.999
    normalization_clip = True

    # Model parameters
    rnn_dims = 680
    fc_dims = 512
    pad = 2

    upsample_factors = (
        4, 4, 16
    )  # Upsample factors must multiply to be equal top hop_size: 4x4x16 = 256

    compute_dims = 128
    res_out_dims = 128
    res_blocks = 10

    # Training parameters
    batch_size = 16
    num_epochs = 2000
    save_every = 10000
    eval_every = 10000

    seq_len_factor = 5
    seq_len = seq_len_factor * hop_size

    grad_norm = 10

    init_lr = 1e-3
    lr_schedule = "step"
    step_gamma = 0.5
    lr_step_interval = 15000

    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8
    amsgrad = False
    weight_decay = 0.0
    fix_lr = None
