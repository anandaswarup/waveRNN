# Recurrent Neural Network based Neural Vocoders

PyTorch implementation of waveRNN based neural vocoder, which predicts a raw waveform from a mel-spectrogram. 

## Getting started
### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/

### 1. Preprocessing

```
python preprocess.py --cfg_file <Path to the configuration file> \
                     --dataset_dir <Path to the dataset dir> \
                     --out_dir <Path to the output dir>
```
Currently the preprocessing script supports the following datasets:
- LJSpeech (en, single speaker)

After running the preprocessing script, the following directory structure will be created in the output dir
```
LJSpeech
├── eval.txt
├── mel
│   ├── LJ001-0001.npy
│   ├── LJ001-0002.npy
│   ├── LJ001-0003.npy
│   ├── LJ001-0004.npy
│   ├── LJ001-0005.npy
│   ├── .
│   └── .
├── qwav
│   ├── LJ001-0001.npy
│   ├── LJ001-0002.npy
│   ├── LJ001-0003.npy
│   ├── LJ001-0004.npy
│   ├── LJ001-0005.npy
│   ├── .
│   └── .
└── train.txt
```
train.txt and eval.txt contain the list of file ids to be used for training and eval respectively. `mel/` and `qwav/` 
directories contain the mel-spectrograms and 10 bit mu-law quantized wav files respectively, to be used for training
the WaveRNN model.

### 2. Configuration / Hyperparameters

All configuration parameters / hyperparameters are specified in a json format, and can be found in `config.json`
### 3. Training

```
python train.py --cfg_file <Path to the configuration file (in json format)>  \
                --data_dir <Path to the data dir (the output dir created during the preprocessing step)> \
                --checkpoints_dir <Path to the dir where the training checkpoints will be saved> \ 
                --logs_dir <Path to the dir where the training logs will be written> \
                --checkpoint_path <(Optional) If specified load checkpoint and restart training from that point>
```

Running this model will create `checkpoints_dir` and `logs_dir` at the specified paths, and all training checkpoints
and training logs will be written to the corresponding directories

### 4. Generation

COMING SOON

## Acknowledgements

The code in this repository is based on the code in the following repositories
1. [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN)
2. [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
3. [bshall/UniversalVocoding](https://github.com/bshall/UniversalVocoding)

## References
1. [arXiv:1802.08435](https://arxiv.org/pdf/1802.08435.pdf): Efficient Neural Audio Synthesis
2. [arXiv:1811.06292v2](https://arxiv.org/pdf/1811.06292.pdf): Towards Achieving Robust Universal Neural Vocoding
