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


def split_by_cc_robust(tokens, cc_id, min_tokens=1, min_cc_interval=0):
    """
    Robust speaker splitting by <cc> tokens with safety mechanisms.
    
    Design principles:
    1. Consecutive <cc> tokens are treated as a single speaker change
    2. Leading/trailing <cc> tokens are ignored
    3. Minimum token count constraint (prevents empty speakers)
    4. Fallback: if one speaker is empty, assign all tokens to speaker1
    
    Args:
        tokens: List of token IDs
        cc_id: Token ID for <cc>
        min_tokens: Minimum tokens required per speaker (default: 1)
        min_cc_interval: Minimum tokens between <cc> switches (default: 0, disabled)
    
    Returns:
        spk1_tokens: List of token IDs for speaker 1
        spk2_tokens: List of token IDs for speaker 2
    """
    spk = [[], []]
    cur = 0
    last_was_cc = True  # Treat BOS as cc to ignore leading cc
    
    for tok_id in tokens:
        if tok_id == cc_id:
            # Ignore consecutive cc
            if last_was_cc:
                continue
            
            # Check minimum interval constraint
            if min_cc_interval > 0 and len(spk[cur]) < min_cc_interval:
                continue
            
            # Switch speaker
            cur = 1 - cur
            last_was_cc = True
        else:
            spk[cur].append(tok_id)
            last_was_cc = False
    
    # Post-processing / safety net
    # If one speaker is empty or too short, fallback to all tokens in speaker1
    if len(spk[0]) < min_tokens or len(spk[1]) < min_tokens:
        # Fallback: all tokens to speaker1
        return tokens, []
    
    return spk[0], spk[1]


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
    
    total_samples_count = len(samples)
    print(f"Evaluating {total_samples_count} samples from {jsonl_path}")
    
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
        pbar = tqdm(samples, desc="Evaluating", total=len(samples))
        for sample in pbar:
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
                        
                        # Debug: check logits distribution for first few steps
                        if total_samples < 2 and step < 5:
                            top_k_values, top_k_indices = torch.topk(last_logits[0], k=5)
                            print(f"  Step {step}: top 5 tokens: {[(idx.item(), val.item(), tokenizer.id_to_piece(idx.item())) for idx, val in zip(top_k_indices, top_k_values)]}")
                        
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
                
                # Get cc_id for speaker segmentation
                cc_id = model.cc_id
                
                # Debug: print first sample for verification
                if total_samples == 0:
                    print(f"\nDebug - First sample prediction:")
                    print(f"  Total tokens: {len(pred_tokens)}")
                    print(f"  Token IDs (first 50): {pred_tokens[:50]}")
                    print(f"  Token IDs (last 50): {pred_tokens[-50:]}")
                    print(f"  <cc> token ID: {cc_id}")
                    cc_count = sum(1 for t in pred_tokens if t == cc_id)
                    print(f"  <cc> tokens found: {cc_count}")
                    # Show token IDs around <cc> tokens
                    if cc_count > 0:
                        print(f"  Token IDs around <cc> tokens:")
                        for i, tok_id in enumerate(pred_tokens):
                            if tok_id == cc_id:
                                start = max(0, i-3)
                                end = min(len(pred_tokens), i+4)
                                context = pred_tokens[start:end]
                                print(f"    Position {i}: {context}")
                
                # Filter out special tokens first
                filtered_tokens = []
                for tok_id in pred_tokens:
                    if tok_id in [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.mask_id]:
                        continue
                    filtered_tokens.append(tok_id)
                
                # Debug: print token analysis for first few samples
                if total_samples < 3:
                    print(f"\nDebug - Token analysis for sample {total_samples + 1}:")
                    print(f"  Total tokens (after filtering): {len(filtered_tokens)}")
                    print(f"  First 20 token IDs: {filtered_tokens[:20]}")
                    print(f"  First 20 token pieces: {[tokenizer.id_to_piece(t) for t in filtered_tokens[:20]]}")
                    vocab_size = tokenizer.get_vocab_size()
                    print(f"  <cc> token ID: {cc_id}, piece: {tokenizer.id_to_piece(cc_id) if cc_id < vocab_size else 'N/A'}")
                    cc_count = sum(1 for t in filtered_tokens if t == cc_id)
                    print(f"  <cc> tokens found: {cc_count} ({cc_count/len(filtered_tokens)*100:.1f}% if len(filtered_tokens) > 0 else 0%)")
                    
                    # Print raw hypothesis (with <cc> tokens visible) for sanity check
                    raw_hyp_pieces = [tokenizer.id_to_piece(t) for t in filtered_tokens]
                    raw_hyp_text = ' '.join(raw_hyp_pieces[:50])  # First 50 tokens
                    print(f"  Raw hypothesis (first 50 tokens): {raw_hyp_text}{'...' if len(filtered_tokens) > 50 else ''}")
                    
                    # Print full decoded text without speaker splitting
                    full_text = tokenizer.decode(filtered_tokens).strip()
                    print(f"  Full decoded text (no split): '{full_text[:200]}{'...' if len(full_text) > 200 else ''}'")
                
                # Robust speaker splitting with safety mechanisms
                spk1_tokens, spk2_tokens = split_by_cc_robust(
                    filtered_tokens, 
                    cc_id, 
                    min_tokens=1,  # Minimum 1 token per speaker
                    min_cc_interval=0  # No minimum interval (can adjust if needed)
                )
                
                # Check for pathological case: too many <cc> tokens
                cc_ratio = sum(1 for t in filtered_tokens if t == cc_id) / len(filtered_tokens) if len(filtered_tokens) > 0 else 0
                if cc_ratio > 0.3:
                    if total_samples < 3:
                        print(f"  Warning: High <cc> ratio ({cc_ratio:.2%}), decoding may be unreliable")
                    # Fallback: all tokens to speaker1
                    spk1_tokens, spk2_tokens = filtered_tokens, []
                
                # Decode speaker segments
                pred_segments = []
                if spk1_tokens:
                    spk1_text = tokenizer.decode(spk1_tokens).strip()
                    pred_segments.append(spk1_text)
                else:
                    pred_segments.append("")
                
                if spk2_tokens:
                    spk2_text = tokenizer.decode(spk2_tokens).strip()
                    pred_segments.append(spk2_text)
                else:
                    pred_segments.append("")
                
                # Debug: print segment details for first few samples
                if total_samples < 3:
                    print(f"  Speaker 1: {len(spk1_tokens)} tokens -> '{pred_segments[0][:100]}{'...' if len(pred_segments[0]) > 100 else ''}'")
                    print(f"  Speaker 2: {len(spk2_tokens)} tokens -> '{pred_segments[1][:100]}{'...' if len(pred_segments[1]) > 100 else ''}'")
                
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
                
                # Ensure pred_segments has exactly 2 elements (speaker1, speaker2)
                # This is guaranteed by split_by_cc_robust, but add safety check
                while len(pred_segments) < 2:
                    pred_segments.append("")
                
                # For 2-speaker case, use first 2 segments
                if num_ref_speakers == 2:
                    pred_segments = pred_segments[:2]
                elif num_ref_speakers > 2:
                    # Pad with empty strings if more than 2 speakers
                    pred_segments.extend([''] * (num_ref_speakers - len(pred_segments)))
                    if total_samples < 2:
                        print(f"  Warning: Reference has {num_ref_speakers} speakers, padding predictions")
                else:
                    # Single speaker case: use first segment only
                    pred_segments = [pred_segments[0]] if pred_segments else [""]
                
                # Compute cpWER following the standard procedure:
                # 1. Concatenate all utterances of each speaker (ref_texts and pred_segments are already per-speaker)
                # 2. Compute WER for all possible speaker permutations
                # 3. Pick the lowest WER
                cpwer = compute_cpwer(ref_texts, pred_segments)
                
                # Print prediction results for each sample
                print(f"\n{'='*80}")
                print(f"Sample {total_samples + 1}/{total_samples_count}: cpWER = {cpwer:.4f} ({cpwer*100:.2f}%)")
                print(f"{'-'*80}")
                for spk_idx, (ref_text, pred_text) in enumerate(zip(ref_texts, pred_segments)):
                    print(f"Speaker {spk_idx + 1}:")
                    print(f"  Reference: {ref_text}")
                    print(f"  Predicted: {pred_text}")
                if len(pred_segments) > len(ref_texts):
                    for spk_idx in range(len(ref_texts), len(pred_segments)):
                        print(f"Speaker {spk_idx + 1} (extra):")
                        print(f"  Predicted: {pred_segments[spk_idx]}")
                print(f"{'='*80}")
                
                total_cpwer += cpwer
                total_samples += 1
                
                # Update progress bar with current and average cpWER
                pbar.set_postfix({
                    'cpWER': f'{cpwer*100:.1f}%',
                    'avg_cpWER': f'{total_cpwer/total_samples*100:.1f}%',
                    'samples': f'{total_samples}/{total_samples_count}'
                })
                
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
