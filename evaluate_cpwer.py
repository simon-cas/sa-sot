#!/usr/bin/env python3
"""
Evaluate cpWER (Character-level Word Error Rate) for multi-speaker ASR

Copyright (c) 2026 Simon Fang
Email: fangshuming519@gmail.com

MIT License
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import editdistance
from itertools import permutations


def compute_cpwer(ref_texts, hyp_texts):
    """
    Compute Character-level Word Error Rate (cpWER) for multi-speaker ASR
    
    The cpWER is computed as follows:
    1. Concatenate all utterances of each speaker for both reference and hypothesis
    2. Compute the WER between the reference and all possible speaker permutations of the hypothesis
    3. Pick the lowest WER among them (best permutation)
    
    Args:
        ref_texts: List of reference texts, one per speaker
        hyp_texts: List of hypothesis texts, one per speaker
    
    Returns:
        cpwer: Character-level WER (float)
    """
    # Step 1: Concatenate all utterances of each speaker
    ref_concatenated = ' '.join(ref_texts).lower()
    hyp_concatenated = ' '.join(hyp_texts).lower()
    
    if len(ref_concatenated) == 0:
        if len(hyp_concatenated) == 0:
            return 0.0
        else:
            return 1.0
    
    # Step 2: Compute WER for all possible speaker permutations
    # Step 3: Pick the lowest WER
    num_speakers = len(ref_texts)
    min_cpwer = float('inf')
    
    # Generate all permutations of hypothesis speaker indices
    for perm in permutations(range(num_speakers)):
        # Reorder hypothesis texts according to permutation
        permuted_hyp_texts = [hyp_texts[i] for i in perm]
        hyp_permuted = ' '.join(permuted_hyp_texts).lower()
        
        # Compute character-level edit distance
        edit_dist = editdistance.eval(ref_concatenated, hyp_permuted)
        
        # cpWER = edit_distance / reference_length
        cpwer = edit_dist / len(ref_concatenated)
        min_cpwer = min(min_cpwer, cpwer)
    
    return min_cpwer


def evaluate_jsonl(jsonl_path, model, tokenizer, device, config):
    """
    Evaluate model on LibriSpeechMix format JSONL file
    
    Args:
        jsonl_path: Path to JSONL file
        model: Trained model
        tokenizer: BPE tokenizer
        device: Device to use
        config: Configuration dict
    
    Returns:
        avg_cpwer: Average cpWER across all samples
    """
    import torch
    import torchaudio
    from data.librispeech import LibriSpeechMixDataset
    
    model.eval()
    
    # Load samples
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Evaluating {len(samples)} samples from {jsonl_path}")
    
    total_cpwer = 0.0
    total_samples = 0
    
    # Audio processing parameters
    sample_rate = config['data']['sample_rate']
    n_mels = config['data']['n_mels']
    hop_length = config['data']['hop_length']
    win_length = config['data']['win_length']
    n_fft = config['data']['n_fft']
    
    # Mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating"):
            try:
                # Load audio
                audio_path = Path(jsonl_path).parent / sample['mixed_wav']
                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue
                
                waveform, sr = torchaudio.load(str(audio_path))
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Extract features
                feats = mel_transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)
                feats = (feats + 1e-8).log()
                feats = feats.unsqueeze(0).to(device)  # (1, T, n_mels)
                feat_lens = torch.tensor([feats.shape[1]], dtype=torch.long).to(device)
                
                # Forward pass - get CIF output length first
                batch_size = feats.shape[0]
                
                # Run encoder and CIF to get token length
                h = model.asr_encoder(feats, feat_lens)
                we = model.weight_estimator(h.transpose(1, 2)).transpose(1, 2)
                c, token_lens, boundaries = model.cif(h, we, feat_lens)
                
                # Get token length for this sample
                token_len = token_lens[0].item()
                if token_len == 0:
                    token_len = 100  # Fallback
                
                # Create prev_tokens with correct length
                prev_tokens = torch.zeros(batch_size, token_len, dtype=torch.long).to(device)
                
                # Forward pass
                output = model(feats, feat_lens, prev_tokens=prev_tokens, 
                             target_len=token_lens, return_intermediate=True)
                
                # Get token predictions (greedy decoding)
                # Output is a dict with 'asr_logits' key
                if isinstance(output, dict):
                    token_logits = output.get('asr_logits', output.get('token_logits'))
                else:
                    # If return_intermediate=False, output is directly the logits
                    token_logits = output
                
                if token_logits is None:
                    print(f"Warning: No logits found in output, skipping sample")
                    continue
                
                # Handle different output formats
                if isinstance(token_logits, dict):
                    # SAT format - use first speaker's logits
                    token_logits = list(token_logits.values())[0]
                
                # Ensure token_logits is 3D: (B, N, V)
                if token_logits.dim() == 4:
                    # If 4D, take first element or reshape
                    token_logits = token_logits.squeeze(0) if token_logits.shape[0] == 1 else token_logits.view(-1, token_logits.shape[-2], token_logits.shape[-1])
                elif token_logits.dim() == 2:
                    # If 2D, add batch dimension
                    token_logits = token_logits.unsqueeze(0)
                
                token_preds = token_logits.argmax(dim=-1)  # (B, N)
                
                # Parse tokens to extract speaker segments
                # In t-SOT format: [spk1_tokens] <cc> [spk2_tokens] <cc> ...
                # We need to split by <cc> tokens at token level, not by spaces in text
                pred_tokens = token_preds[0].cpu().tolist()
                pred_segments = []
                current_segment_tokens = []
                cc_id = model.cc_id
                
                for tok_id in pred_tokens:
                    if tok_id == tokenizer.pad_id or tok_id == tokenizer.bos_id or tok_id == tokenizer.eos_id:
                        continue
                    elif tok_id == tokenizer.mask_id:
                        continue
                    elif tok_id == cc_id:
                        # <cc> marks speaker change, decode current segment and start new one
                        if current_segment_tokens:
                            # Decode current segment tokens to text
                            segment_text = ''.join([tokenizer.id_to_piece(t) for t in current_segment_tokens])
                            pred_segments.append(segment_text.strip())
                            current_segment_tokens = []
                    else:
                        # Regular token, add to current segment
                        current_segment_tokens.append(tok_id)
                
                # Don't forget the last segment (if no <cc> at the end)
                if current_segment_tokens:
                    segment_text = ''.join([tokenizer.id_to_piece(t) for t in current_segment_tokens])
                    pred_segments.append(segment_text.strip())
                
                # Get reference text from 'texts' field (LibriSpeechMix format)
                # texts is a list of transcriptions for each speaker
                ref_texts = sample.get('texts', [])
                if not ref_texts:
                    # Fallback: try to get from speaker_profiles
                    ref_texts = []
                    for spk_info in sample.get('speaker_profiles', []):
                        ref_text = spk_info.get('transcription', '').strip()
                        if ref_text:
                            ref_texts.append(ref_text)
                
                if not ref_texts:
                    continue
                
                # For cpWER with multiple speakers, we need to find the best permutation
                # that minimizes the character-level edit distance
                # This handles the case where model output order might differ from reference order
                num_ref_speakers = len(ref_texts)
                num_pred_speakers = len(pred_segments)
                
                # If number of speakers don't match, pad or truncate
                if num_pred_speakers < num_ref_speakers:
                    # Pad with empty strings
                    pred_segments.extend([''] * (num_ref_speakers - num_pred_speakers))
                elif num_pred_speakers > num_ref_speakers:
                    # Truncate to match reference
                    pred_segments = pred_segments[:num_ref_speakers]
                
                # Compute cpWER following the standard procedure:
                # 1. Concatenate all utterances of each speaker (ref_texts and pred_segments are already per-speaker)
                # 2. Compute WER for all possible speaker permutations
                # 3. Pick the lowest WER
                cpwer = compute_cpwer(ref_texts, pred_segments)
                total_cpwer += cpwer
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if total_samples == 0:
        return None
    
    avg_cpwer = total_cpwer / total_samples
    return avg_cpwer


def main():
    parser = argparse.ArgumentParser(description='Evaluate cpWER on LibriSpeechMix dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--jsonl', type=str, required=True,
                       help='Path to JSONL evaluation file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    import yaml
    import torch
    from model.sasot_model import SASOTModel
    from data.tokenizer import BPETokenizer
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device)
    
    # Load tokenizer
    tokenizer_path = config['tokenizer']['model_path']
    tokenizer = BPETokenizer(tokenizer_path)
    
    # Load model
    model = SASOTModel(
        vocab_size=config['model']['vocab_size'],
        num_speakers=config['model']['num_speakers'],
        asr_dim=config['model']['asr_dim'],
        spk_dim=config['model']['spk_dim'],
        cif_beta=config['model']['cif_beta']
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Evaluate
    print(f"Evaluating on {args.jsonl}")
    avg_cpwer = evaluate_jsonl(args.jsonl, model, tokenizer, device, config)
    
    if avg_cpwer is not None:
        print(f"\n{'='*60}")
        print(f"Average cpWER: {avg_cpwer:.4f} ({avg_cpwer*100:.2f}%)")
        print(f"{'='*60}")
    else:
        print("Evaluation failed - no samples processed")


if __name__ == '__main__':
    main()
