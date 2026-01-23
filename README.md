# SA-SOT

Unofficial PyTorch implementation of SA-SOT: Speaker-Aware Serialized Output Training for Multi-Talker ASR.

## Description

This repository implements the SA-SOT model for multi-talker automatic speech recognition (ASR). SA-SOT extends Serialized Output Training (SOT) with speaker-aware mechanisms to better handle overlapped speech from multiple speakers.

## Setup

Activate the environment:
```bash
conda activate sa-sot
```

Install required packages:
```bash
pip install -r requirements.txt
```

**Note**: PyTorch and TorchAudio versions should match your CUDA version. If you need specific versions, install them separately:
```bash
# Example for CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu12.4
pip install -r requirements.txt
```

## Quick Start

Edit `config.yaml` to set your dataset paths, then run:

```bash
bash run.sh
```

The script will automatically handle data preparation, training, and evaluation.

## Configuration

Edit `config.yaml` to set:
- Dataset paths (`data.librispeech_root`)
- Training parameters (`training.batch_size`, `training.num_epochs`, etc.)
- Model parameters (`model.vocab_size`, `model.num_speakers`, etc.)

You can also modify default parameters at the top of `run.sh` (GPU IDs, config file path, etc.).

## License

MIT License

Copyright (c) 2026 Simon Fang

Email: fangshuming519@gmail.com

See [LICENSE](LICENSE) file for details.
