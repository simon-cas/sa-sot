#!/bin/bash

# Copyright (c) 2026 Simon Fang
# Email: fangshuming519@gmail.com
# MIT License

set -e

stage=6
stop_stage=7

data=data
exp=exp
exp_name=sasot
gpus="1 2"
config=config.yaml
skip_alignment=false
skip_eval=false
single_speaker=false
eval_checkpoint=""

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

# Determine Python command to use
# Priority: 1) Environment variable PYTHON_CMD, 2) Current conda env, 3) python3, 4) python
if [ -n "$PYTHON_CMD" ]; then
    # Use explicitly set PYTHON_CMD
    :
elif [ -n "$CONDA_DEFAULT_ENV" ] && command -v python &> /dev/null; then
    # Use current conda environment's Python
    PYTHON_CMD=$(command -v python)
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    # Use conda prefix Python if available
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
elif command -v python3 &> /dev/null; then
    # Use python3 if available
    PYTHON_CMD=$(command -v python3)
elif command -v python &> /dev/null; then
    # Fall back to python
    PYTHON_CMD=$(command -v python)
else
    echo "Error: Python not found. Please install Python or set PYTHON_CMD environment variable."
    echo "  Example: PYTHON_CMD=/path/to/python ./run.sh"
    echo "  Or activate your conda environment: conda activate sa-sot"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Set random seed for reproducibility
export PYTHONHASHSEED=1234

# Convert space-separated GPU IDs to comma-separated for CUDA_VISIBLE_DEVICES
# e.g., "1 2" -> "1,2"
cuda_visible_devices=$(echo $gpus | tr ' ' ',')
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices

# Read paths from config.yaml
librispeech_root=$($PYTHON_CMD -c "import yaml; f=open('$config'); c=yaml.safe_load(f); print(c['data']['librispeech_root'])" 2>/dev/null || echo "")
train_manifest=$($PYTHON_CMD -c "import yaml; f=open('$config'); c=yaml.safe_load(f); print(c['data']['train_manifest'])" 2>/dev/null || echo "$data/manifests/train_manifest.json")
output_dir=$(dirname $train_manifest)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage1: Preparing training manifests..."
  if [ -z "$librispeech_root" ]; then
    echo "Error: librispeech_root not found in config.yaml"
    exit 1
  fi
  
      $PYTHON_CMD data/prepare_manifest.py \
    --root_dir $librispeech_root \
    --output_dir $output_dir \
    --mix_samples 0
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage2: Training BPE tokenizer..."
        tokenizer_dir=$(dirname $($PYTHON_CMD -c "import yaml; f=open('$config'); c=yaml.safe_load(f); print(c['tokenizer']['model_path'])" 2>/dev/null || echo "$data/tokenizer/bpe_tokenizer.model"))
  
      $PYTHON_CMD data/train_bpe_tokenizer.py \
    --manifest $train_manifest \
    --output_dir $tokenizer_dir \
    --vocab_size 3730 \
    --character_coverage 0.9995 \
    --model_type bpe
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  if [ "$skip_alignment" = "false" ]; then
    echo "Stage3: Running forced alignment..."
        alignment_path=$($PYTHON_CMD -c "import yaml; f=open('$config'); c=yaml.safe_load(f); print(c['alignment']['word_alignments_path'])" 2>/dev/null || echo "$data/alignments/word_alignments.json")
        alignment_dir=$(dirname $alignment_path)
        mkdir -p $alignment_dir

        # Use first GPU from gpus parameter
        first_gpu=$(echo $gpus | awk '{print $1}')
        alignment_device="cuda:$first_gpu"
        echo "Using device for alignment: $alignment_device"

        $PYTHON_CMD data/prepare_torchaudio_alignment.py \
      --manifest $train_manifest \
      --output $alignment_path \
      --device $alignment_device
  else
    echo "Stage3: Skipping forced alignment (--skip-alignment)"
  fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  if [ "$skip_eval" = "false" ]; then
    echo "Stage4: Preparing LibriSpeechMix evaluation dataset..."
    if [ -z "$librispeech_root" ]; then
      echo "Error: librispeech_root not found in config.yaml"
      exit 1
    fi
    
        librispeechmix_dir=$($PYTHON_CMD -c "import yaml; f=open('$config'); c=yaml.safe_load(f); print(c.get('data', {}).get('librispeechmix_dir', '$data/librispeechmix'))" 2>/dev/null || echo "$data/librispeechmix")

        $PYTHON_CMD data/prepare_librispeechmix.py \
      --root_dir $librispeech_root \
      --output_dir $librispeechmix_dir \
      --splits dev-clean test-clean \
      --min_overlap 0.1 \
      --max_overlap 0.5
  else
    echo "Stage4: Skipping LibriSpeechMix preparation (--skip-eval)"
  fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage5: Verifying configuration..."
  echo "Config file: $config"
  echo "Experiment directory: $exp_dir"
  echo "Training manifest: $train_manifest"
  echo "LibriSpeech root: $librispeech_root"
  echo "GPUs: $gpus"
  echo "Random seed: 1234"
  echo "On-the-fly mixing: $([ "$single_speaker" = "true" ] && echo "disabled" || echo "enabled")"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage6: Training the SA-SOT model..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  
  if [ "$single_speaker" = "true" ]; then
    use_mix_flag=""
  else
    use_mix_flag="--use_mix"
  fi
  
  if [ $num_gpu -eq 1 ]; then
    $PYTHON_CMD train.py --config $config $use_mix_flag \
           --seed 1234 \
           --train_manifest $train_manifest \
           --exp_dir $exp_dir
  else
    # Use python -m torch.distributed.run instead of torchrun for better compatibility
    $PYTHON_CMD -m torch.distributed.run --nproc_per_node=$num_gpu train.py --config $config $use_mix_flag \
             --seed 1234 \
             --train_manifest $train_manifest \
             --exp_dir $exp_dir
  fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage7: Evaluating cpWER on test sets..."
  
  # Determine checkpoint path
  if [ -n "$eval_checkpoint" ]; then
    checkpoint_path="$eval_checkpoint"
  else
    # Auto-detect checkpoint: use the last epoch checkpoint (highest epoch number)
    # Find the checkpoint with the highest epoch number
    latest_checkpoint=$(ls -t $exp_dir/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$latest_checkpoint" ]; then
      checkpoint_path="$latest_checkpoint"
    elif [ -f "$exp_dir/final_model.pt" ]; then
      checkpoint_path="$exp_dir/final_model.pt"
    else
      echo "Error: No checkpoint found. Please specify --eval-checkpoint or ensure training completed."
      exit 1
    fi
  fi
  
  if [ ! -f "$checkpoint_path" ]; then
    echo "Error: Checkpoint not found: $checkpoint_path"
    exit 1
  fi
  
  echo "Using checkpoint: $checkpoint_path"
  
  # Determine evaluation device from gpus parameter (use first GPU)
  first_gpu=$(echo $gpus | awk '{print $1}')
  eval_device="cuda:$first_gpu"
  echo "Using evaluation device: $eval_device"
  
  # Get LibriSpeechMix directory
  librispeechmix_dir=$($PYTHON_CMD -c "import yaml; f=open('$config'); c=yaml.safe_load(f); print(c.get('data', {}).get('librispeechmix_dir', '$data/librispeechmix'))" 2>/dev/null || echo "$data/librispeechmix")
  
  # Evaluate on dev-clean-2mix.jsonl
  dev_jsonl="$librispeechmix_dir/dev-clean-2mix.jsonl"
  if [ -f "$dev_jsonl" ]; then
    echo "Evaluating on dev-clean-2mix.jsonl..."
    $PYTHON_CMD evaluate_cpwer.py \
      --checkpoint "$checkpoint_path" \
      --jsonl "$dev_jsonl" \
      --config "$config" \
      --device "$eval_device" \
      2>&1 | tee "$exp_dir/eval_dev_cpwer.log"
  else
    echo "Warning: $dev_jsonl not found. Skipping dev-clean evaluation."
  fi
  
  # Evaluate on test-clean-2mix.jsonl
  test_jsonl="$librispeechmix_dir/test-clean-2mix.jsonl"
  if [ -f "$test_jsonl" ]; then
    echo "Evaluating on test-clean-2mix.jsonl..."
    $PYTHON_CMD evaluate_cpwer.py \
      --checkpoint "$checkpoint_path" \
      --jsonl "$test_jsonl" \
      --config "$config" \
      --device "$eval_device" \
      2>&1 | tee "$exp_dir/eval_test_cpwer.log"
  else
    echo "Warning: $test_jsonl not found. Skipping test-clean evaluation."
  fi
  
  echo "Stage7: Evaluation completed. Results saved in $exp_dir/eval_*.log"
fi
