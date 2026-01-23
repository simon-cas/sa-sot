#!/usr/bin/env python3
"""
Prepare word alignments using PyTorch Audio's forced alignment (Wav2Vec2 + CTC).

This script uses torchaudio's Wav2Vec2 model and CTC segmentation to align
transcripts with audio, generating word emission times for SA-SOT training.

Reference: https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label} ({self.score:.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def get_trellis(emission, tokens, blank_id=0):
    """
    Generate trellis matrix for CTC alignment.
    
    Args:
        emission: Frame-wise label probability (T, num_labels)
        tokens: Token sequence to align (N,)
        blank_id: ID of blank token in CTC
        
    Returns:
        trellis: Trellis matrix (T+1, N+1)
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    
    # Trellis has extra dimension for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for transition of blank token
    trellis = torch.zeros((num_frame + 1, num_tokens + 1))
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for advancing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    """
    Backtrack to find the most likely path through the trellis.
    
    Returns:
        path: List of (time_index, token_index) pairs
    """
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()
    
    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        
        # Store path segment
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        path.append((t - 1, j - 1, prob))
        
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


def merge_repeats(path, tokens, labels):
    """
    Merge repeated labels in the path.
    
    Args:
        path: List of (time_index, token_index, prob) tuples from backtrack
        tokens: Token sequence
        labels: Label vocabulary
        
    Returns:
        segments: List of Segment with merged repeats
    """
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        if i2 >= len(path) or path[i1][1] != path[i2][1]:
            if i1 != i2:
                segs = path[i1:i2]
                token_idx = segs[0][1]
                segments.append(Segment(
                    label=labels[tokens[token_idx]],
                    start=segs[0][0],
                    end=segs[-1][0] + 1,
                    score=sum(seg[2] for seg in segs) / len(segs)
                ))
            i1 = i2
        else:
            i2 += 1
    return segments


def merge_words(segments, separator="|"):
    """
    Merge character segments into words.
    
    Args:
        segments: List of character-level segments
        separator: Word separator token
        
    Returns:
        words: List of word-level segments
    """
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(
                    word,
                    segments[i1].start,
                    segments[i2 - 1].end,
                    score
                ))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def align_utterance(audio_path, transcript, model, labels, device, sample_rate=16000):
    """
    Perform forced alignment for a single utterance.
    
    Args:
        audio_path: Path to audio file
        transcript: Text transcription
        model: Wav2Vec2 model
        labels: Label vocabulary
        device: Device to run on
        sample_rate: Audio sample rate
        
    Returns:
        word_segments: List of word segments with timing information
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    waveform = waveform.to(device)
    
    # Get frame-wise emissions
    with torch.inference_mode():
        emissions, _ = model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)
    
    emission = emissions[0].cpu().detach()
    
    # Prepare transcript: convert to uppercase and add word separators
    transcript = transcript.upper().strip()
    # Add word separators
    transcript_with_sep = "|".join(transcript.split())
    
    # Convert transcript to token indices
    tokens = [labels.index(c) if c in labels else labels.index('-') for c in transcript_with_sep]
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Get blank token ID (usually 0 or '-')
    blank_id = labels.index('-') if '-' in labels else 0
    
    # Generate trellis
    trellis = get_trellis(emission, tokens, blank_id)
    
    # Backtrack to find path
    path = backtrack(trellis, emission, tokens, blank_id)
    
    # Merge repeats
    segments = merge_repeats(path, tokens, labels)
    
    # Merge into words
    word_segments = merge_words(segments, separator="|")
    
    # Convert frame indices to time (seconds)
    # Wav2Vec2 model outputs at ~50ms per frame (20 frames per second)
    # Calculate actual frame rate from audio duration
    audio_duration = waveform.size(1) / sample_rate
    num_frames = emission.size(0)
    time_per_frame = audio_duration / num_frames if num_frames > 0 else 0.05
    
    word_alignments = []
    for word in word_segments:
        start_time = word.start * time_per_frame
        end_time = word.end * time_per_frame
        duration = end_time - start_time
        
        word_alignments.append({
            'word': word.label,
            'start': start_time,
            'end': end_time,
            'duration': duration
        })
    
    return word_alignments


def main():
    parser = argparse.ArgumentParser(
        description='Generate word alignments using PyTorch Audio forced alignment'
    )
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to manifest JSON file')
    parser.add_argument('--output', type=str, default='./data/alignments/word_alignments.json',
                       help='Output path for word alignments JSON')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing (currently only 1 is supported)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Audio sample rate')
    
    args = parser.parse_args()
    
    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    samples = manifest.get('samples', [])
    print(f"Found {len(samples)} samples")
    
    # Load Wav2Vec2 model
    print("Loading Wav2Vec2 model...")
    device = torch.device(args.device)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    sample_rate = bundle.sample_rate
    
    print(f"Model loaded on {device}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Labels: {len(labels)} characters")
    
    # Process each sample
    word_alignments = {}
    failed_count = 0
    
    print("\nProcessing alignments...")
    for sample in tqdm(samples, desc="Aligning"):
        audio_file = sample['audio_file']
        transcription = sample.get('transcription', '').strip()
        utterance_id = sample.get('id', '')
        
        if not transcription or not os.path.exists(audio_file):
            failed_count += 1
            continue
        
        try:
            # Get utterance ID from audio file path if not in manifest
            if not utterance_id:
                # Extract from path: .../speaker/chapter/file.flac -> speaker-chapter-file
                parts = Path(audio_file).parts
                if len(parts) >= 3:
                    utterance_id = f"{parts[-3]}-{parts[-2]}-{Path(parts[-1]).stem}"
                else:
                    utterance_id = Path(audio_file).stem
            
            # Perform alignment
            alignments = align_utterance(
                audio_file,
                transcription,
                model,
                labels,
                device,
                sample_rate
            )
            
            if alignments:
                word_alignments[utterance_id] = alignments
        except Exception as e:
            print(f"\nError aligning {audio_file}: {e}")
            failed_count += 1
            continue
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(word_alignments, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Processed {len(word_alignments)} utterances successfully")
    if failed_count > 0:
        print(f"  Failed: {failed_count} utterances")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()

