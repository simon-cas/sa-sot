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

**Note**: 
- PyTorch and TorchAudio versions should match your CUDA version. If you need specific versions, install them separately:
  ```bash
  # Example for CUDA 12.4
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu12.4
  pip install -r requirements.txt
  ```
- The forced alignment uses Wav2Vec2 model from torchaudio

**Note**: We use PyTorch Audio's forced alignment (Wav2Vec2 + CTC) instead of MFA to avoid CUDA version conflicts and keep everything in the PyTorch ecosystem. See Step 3 below for alignment setup.

## Complete Setup and Training Workflow

### Step 1: Prepare Training Manifests

Generate manifest files for training data:

```bash
python data/prepare_manifest.py \
    --root_dir /data/work/SimonFang/datasets/english/LibriSpeech \
    --output_dir ./data/manifests \
    --mix_samples 0  # 0 = truly on-the-fly mixing (recommended for reproduction)
```

**Output**: 
- `./data/manifests/train_manifest.json` - Single speaker training data
- `./data/manifests/dev_manifest.json` - Validation data
- `./data/manifests/test_manifest.json` - Test data
- `./data/manifests/mix_train_manifest.json` - Mix configurations for on-the-fly training

**Note**: 
- `--mix_samples 0` (default): Truly on-the-fly mixing - random mixing each epoch (recommended for paper reproduction)
- `--mix_samples > 0`: Pre-generate mixing configurations for reproducibility (not required by paper)

### Step 2: Train BPE Tokenizer

Train SentencePiece BPE tokenizer (required for proper tokenization):

```bash
python data/train_bpe_tokenizer.py \
    --manifest ./data/manifests/train_manifest.json \
    --output_dir ./data/tokenizer \
    --vocab_size 3730 \
    --character_coverage 0.9995 \
    --model_type bpe
```

**Output**: `./data/tokenizer/bpe_tokenizer.model`

The tokenizer automatically includes special tokens:
- `<cc>` - Speaker change token (for t-SOT format)
- `<mask>` - Mask token (for SAT training)
- `<unk>`, `<s>`, `</s>` - Standard tokens

### Step 3: Forced Alignment (Optional but Recommended)

Get word emission times using PyTorch Audio's forced alignment (Wav2Vec2 + CTC):

```bash
python data/prepare_torchaudio_alignment.py \
    --manifest ./data/manifests/train_manifest.json \
    --output ./data/alignments/word_alignments.json \
    --device cuda  # Use 'cpu' if no GPU available
```

**Output**: `./data/alignments/word_alignments.json` with word emission times

**Note**: 
- This uses `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` model for alignment
- The alignment process may take some time depending on dataset size
- GPU is recommended for faster processing
- Reference: [PyTorch Audio Forced Alignment Tutorial](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)

### Step 4: Prepare LibriSpeechMix Evaluation Dataset

Generate LibriSpeechMix format evaluation sets for model evaluation:

```bash
python data/prepare_librispeechmix.py \
    --root_dir /data/work/SimonFang/datasets/english/LibriSpeech \
    --output_dir ./data/librispeechmix \
    --splits dev-clean test-clean \
    --min_overlap 0.1 \
    --max_overlap 0.5
```

**Output**: 
- `dev-clean-1mix.jsonl`, `dev-clean-2mix.jsonl`, `dev-clean-3mix.jsonl`
- `test-clean-1mix.jsonl`, `test-clean-2mix.jsonl`, `test-clean-3mix.jsonl`
- Corresponding mixed audio files in subdirectories

### Step 5: Configure Training

Edit `config.yaml` to set paths:

```yaml
# Tokenizer
tokenizer:
  model_path: "./data/tokenizer/bpe_tokenizer.model"
  vocab_size: 3730

# Forced Alignment (optional)
alignment:
  word_alignments_path: "./data/alignments/word_alignments.json"  # Word emission times from forced alignment
  use_alignment: true  # Set to false if not using word alignments

# Data
data:
  librispeech_root: "/data/work/SimonFang/datasets/english/LibriSpeech"
  train_manifest: "./data/manifests/train_manifest.json"
  mix_train_manifest: "./data/manifests/mix_train_manifest.json"
  
# Training
training:
  device: "cuda:3"  # Specify GPU
  batch_size: 4
  num_epochs: 10
  learning_rate: 1.0e-4
```

### Step 6: Start Training

**Option A: Use the automated training script (Recommended)**

We provide a `run.sh` script that automates all steps from data preparation to training. You can modify the default parameters at the top of the script, or override them via command line.

```bash
# Run complete pipeline (all steps, stage 1-6)
./run.sh

# Run specific stages
./run.sh --stage=2                    # Start from step 2
./run.sh --stop_stage=3                # Run steps 1-3
./run.sh --stage=2 --stop_stage=4      # Run steps 2-4

# Skip optional steps
./run.sh --skip-alignment              # Skip forced alignment (step 3)
./run.sh --skip-eval                    # Skip LibriSpeechMix preparation (step 4)

# Training options
./run.sh --single-speaker               # Train with single speaker data only

# Use different config file
./run.sh --config=./config_custom.yaml

# Combine options
./run.sh --stage=2 --stop_stage=6 --skip-alignment
```

**Modify default parameters in `run.sh`:**
```bash
# Edit these variables at the top of run.sh
stage=1
stop_stage=6
data=data
exp=exp
exp_name=sa-sot
gpus="3"  # GPU device ID(s)
config=./config.yaml
```

**Stage descriptions:**
- **Stage 1**: Prepare training manifests
- **Stage 2**: Train BPE tokenizer
- **Stage 3**: Run forced alignment (optional)
- **Stage 4**: Prepare LibriSpeechMix evaluation dataset (optional)
- **Stage 5**: Verify configuration
- **Stage 6**: Start training

**Examples:**
```bash
# Only prepare data (stages 1-2)
./run.sh --stop_stage=2

# Skip data preparation, start from tokenizer training
./run.sh --stage=2 --stop_stage=2

# Prepare everything except training
./run.sh --stop_stage=5

# Only run training (assumes data is already prepared)
./run.sh --stage=6

# Use multiple GPUs
./run.sh --gpus="0 1 2 3"
```

**Note**: 
- Default parameters are defined at the top of `run.sh` - edit them directly if needed
- The script automatically reads paths and device settings from `config.yaml`:
  - `data.librispeech_root` → LibriSpeech dataset path
  - `training.device` → GPU device (e.g., "cuda:3")
  - `data.train_manifest` → Used to infer output directory
- Command-line arguments override the default values in the script

**Option B: Manual step-by-step training**

Train with on-the-fly mixed data (recommended):
```bash
python train.py --config config.yaml --use_mix
```

Train with single speaker data:
```bash
python train.py --config config.yaml
```

## Training Output

- Checkpoints are saved in `./checkpoints/` directory
- Checkpoints are saved every `save_interval` epochs
- Final model is saved as `checkpoints/final_model.pt`

## Evaluation

After training, evaluate on LibriSpeechMix test sets:

```bash
python evaluate.py \
    --config config.yaml \
    --checkpoint ./checkpoints/final_model.pt \
    --eval_manifest ./data/librispeechmix/test-clean-2mix.jsonl
```

The evaluation computes metrics like cpWER (concatenated minimum-permutation word error rate) for different speaker mixture conditions.

## Key Components

### BPE Tokenization
- Uses SentencePiece for subword tokenization
- Vocabulary size: 3730 (configurable)
- Special tokens: `<cc>` for speaker changes, `<mask>` for SAT

### Forced Alignment
- PyTorch Audio's Wav2Vec2 + CTC forced alignment provides word emission times
- Used for better alignment between audio frames and tokens
- Optional but recommended for improved training
- No external dependencies beyond PyTorch ecosystem

### On-the-Fly Mixing
- Training data is mixed dynamically during training
- No pre-mixed audio files needed
- Each epoch uses different random mixtures
- Recommended for paper reproduction

## Dataset Format

### Training Data
- Single speaker manifests: Standard LibriSpeech format
- Mix manifests: Configurations for on-the-fly mixing (not pre-mixed audio)

### Evaluation Data (LibriSpeechMix)
Follows the format from [NaoyukiKanda/LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix):

- **JSONL files**: Each line contains a JSON object with:
  - `id`: Utterance ID
  - `mixed_wav`: Path to mixed audio file
  - `texts`: List of transcriptions for each speaker
  - `speakers`: List of speaker IDs
  - `delays`: List of delays (in seconds) for each utterance
  - `durations`: List of durations for each original audio
  - `speaker_profile`: Optional speaker profile audio files for SA-ASR
  - `speaker_profile_index`: Optional indices mapping utterances to speaker profiles

## File Structure

After complete setup:

```
sa-sot/
├── data/
│   ├── manifests/
│   │   ├── train_manifest.json
│   │   ├── dev_manifest.json
│   │   ├── test_manifest.json
│   │   └── mix_train_manifest.json
│   ├── tokenizer/
│   │   ├── bpe_tokenizer.model
│   │   └── bpe_tokenizer.vocab
│   ├── alignments/
│   │   └── word_alignments.json  # Word emission times from forced alignment
│   └── librispeechmix/        # Evaluation dataset
│       ├── dev-clean-1mix.jsonl
│       ├── dev-clean-2mix.jsonl
│       ├── dev-clean-3mix.jsonl
│       └── ...
├── tests/                     # Unit tests
├── config.yaml                # Training configuration
└── train.py                   # Training script
```

## Notes

1. **BPE Tokenizer**: Required for proper tokenization. The model uses BPE subword units as specified in the SA-SOT paper.

2. **Forced Alignment**: Optional but recommended. Word emission times from PyTorch Audio's forced alignment can be used for better alignment between audio frames and tokens.

3. **Special Tokens**: 
   - `<cc>` (change-of-character) indicates speaker changes in t-SOT format
   - `<mask>` is used in SAT (Speaker-Aware Training) for masking other speakers' tokens

4. **On-the-Fly Mixing**: Training uses dynamic mixing by default (recommended for paper reproduction). Each epoch generates different random mixtures.

5. **Memory Usage**: If you encounter OOM errors, reduce `batch_size` or `max_duration` in config.

## References

- SA-SOT Paper: Speaker-Aware Serialized Output Training for Multi-Talker ASR
- LibriSpeechMix: [NaoyukiKanda/LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- PyTorch Audio Forced Alignment: [Forced Alignment Tutorial](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)

## License

MIT License

Copyright (c) 2026 Simon Fang

Email: fangshuming519@gmail.com

See [LICENSE](LICENSE) file for details.
