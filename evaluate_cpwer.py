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
                
                # ========= Autoregressive decoding =========
                # Initialize with BOS token
                max_decode_len = min(token_len * 2, 500)  # Allow some extra length for generation
                pred_tokens_list = []
                prev_tokens = torch.full((batch_size, 1), tokenizer.bos_id, dtype=torch.long).to(device)
                
                # Get speaker embeddings (token_spk) from encoder output
                # We need to run the model once to get the intermediate outputs
                with torch.no_grad():
                    # First, get encoder outputs and speaker embeddings
                    h = model.asr_encoder(feats, feat_lens)
                    we = model.weight_estimator(h.transpose(1, 2)).transpose(1, 2)
                    alpha = we.squeeze(-1)
                    c, token_lens_cif, boundaries = model.cif(h, alpha, token_lens)
                    
                    # Get speaker embeddings
                    spk_frame = model.spk_encoder(feats)
                    # Use static method from model's class
                    token_spk = type(model)._boundary_pooling(spk_frame, boundaries)
                    
                    # Autoregressive decoding
                    for step in range(max_decode_len):
                        # Get the current length of prev_tokens
                        current_len = prev_tokens.shape[1]
                        
                        # Ensure token_spk has the right length to match prev_tokens
                        # token_spk is (B, N_cif, D), we need to pad or repeat to match prev_tokens length
                        if current_len <= token_spk.shape[1]:
                            # Use first current_len tokens
                            token_spk_used = token_spk[:, :current_len, :]
                        else:
                            # Pad with the last token_spk value
                            padding_len = current_len - token_spk.shape[1]
                            last_token_spk = token_spk[:, -1:, :]  # (B, 1, D)
                            padding = last_token_spk.repeat(1, padding_len, 1)  # (B, padding_len, D)
                            token_spk_used = torch.cat([token_spk, padding], dim=1)  # (B, current_len, D)
                        
                        # Forward pass with current prev_tokens
                        asr_logits = model.asr_decoder(c, prev_tokens, token_spk_used)  # (B, N, V)
                        
                        # Get logits for the last position
                        last_logits = asr_logits[:, -1, :]  # (B, V)
                        
                        # Greedy decoding: select token with highest probability
                        next_token = last_logits.argmax(dim=-1)  # (B,)
                        next_token_id = next_token[0].item()
                        
                        # Stop if EOS token
                        if next_token_id == tokenizer.eos_id:
                            break
                        
                        # Add to predictions
                        pred_tokens_list.append(next_token_id)
                        
                        # Append to prev_tokens for next step
                        prev_tokens = torch.cat([prev_tokens, next_token.unsqueeze(1)], dim=1)  # (B, N+1)
                
                # Convert to tensor for compatibility with existing code
                token_preds = torch.tensor([pred_tokens_list], dtype=torch.long).to(device)  # (1, N)
                
                # Parse tokens to extract speaker segments
                # In t-SOT format: [spk1_tokens] <cc> [spk2_tokens] <cc> ...
                # We need to split by <cc> tokens at token level, not by spaces in text
                pred_tokens = token_preds[0].cpu().tolist()
                
                # Debug: print first 50 token IDs and their decoded pieces
                if total_samples < 2:
                    print(f"\nDebug - First 50 token IDs: {pred_tokens[:50]}")
                    print(f"Debug - Decoded pieces:")
                    for i, tok_id in enumerate(pred_tokens[:50]):
                        piece = tokenizer.id_to_piece(tok_id)
                        print(f"  [{i}] ID={tok_id}, piece='{piece}'")
                
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
                            # Decode current segment tokens to text (BPE -> words)
                            segment_text = tokenizer.decode(current_segment_tokens)
                            pred_segments.append(segment_text.strip())
                            current_segment_tokens = []
                    else:
                        # Regular token, add to current segment
                        current_segment_tokens.append(tok_id)
                
                # Don't forget the last segment (if no <cc> at the end)
                if current_segment_tokens:
                    # Decode current segment tokens to text (BPE -> words)
                    segment_text = tokenizer.decode(current_segment_tokens)
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
                
                # Debug: print first 2 samples
                print(f"\n{'='*60}")
                print(f"Sample {total_samples + 1}:")
                print(f"  Reference texts ({len(ref_texts)} speakers):")
                for i, ref in enumerate(ref_texts):
                    print(f"    Speaker {i+1}: {ref}")
                print(f"  Predicted segments ({len(pred_segments)} speakers):")
                for i, pred in enumerate(pred_segments):
                    print(f"    Speaker {i+1}: {pred}")
                print(f"  cpWER: {cpwer:.4f} ({cpwer*100:.2f}%)")
                print(f"{'='*60}")
                
                total_cpwer += cpwer
                total_samples += 1
                
                # Exit after 2 samples for debugging
                if total_samples >= 2:
                    print(f"\nExiting after {total_samples} samples for debugging.")
                    break
                
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
