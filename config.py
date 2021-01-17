"""Configuration parameters"""


class Config:

    # Audio processing parameters
    sampling_rate = 22050
    min_db_level = -100
    ref_db_level = 20

    n_fft = 2048
    win_size = 1100  # 50ms window size
    hop_length = 275  # 12.5ms hop length

    n_mels = 80
    fmin = 90

    num_bits = 10  # bit depth of the signal
    mu_law = True
    peak_norm = False  # normalize to the peak of each wav

    # WaveRNN model parameters
    upsample_factors = (
        5, 5, 11)  # upsample factors must correctly factorize hopp length
    rnn_dims = 512
    fc_dims = 512
    compute_dims = 128
    res_out_dims = 128
    num_res_blocks = 10
    output_mode = "mol"  # sample from a mixture of logistics

    # WaveRNN model training
    batch_size = 32
    lr = 1e-4
    num_steps = 1000000  # total number of steps to train the model
    checkpoint_interval = 25000
    num_samples_to_generate_at_checkpoint = 5  # number of samples to generate at each checkpoint
    num_test_samples = 50  # number of samples to hold-out for testing
    pad_factor = 2  # factor by which to pad the input so that the resnet can see wider than input length
    seq_len = hop_length * 5  # this has to be a multiple of hop length
    grad_clip_norm = 4  # gradient clipping

    # WaveRNN generation
    batched_generation = True  # generate in batches
    num_target_samples = 11000  # target number of samples to be generated in each batch entry
    num_overlap_samples = 550  # number of samples for crossfading between batches
