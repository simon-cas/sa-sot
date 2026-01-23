#!/usr/bin/env python3
"""
Prepare LibriSpeechMix evaluation dataset
Generates 1-speaker, 2-speaker-mixed, and 3-speaker-mixed test sets
following the format from https://github.com/NaoyukiKanda/LibriSpeechMix

This script should be run after prepare_manifest.py to generate evaluation sets.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import soundfile as sf
import torch
import torchaudio
from collections import defaultdict


def find_audio_files(root_dir, split_name):
    """Find all audio files in a LibriSpeech split directory"""
    audio_files = []
    split_dir = os.path.join(root_dir, split_name)
    
    if not os.path.exists(split_dir):
        print(f"Warning: {split_dir} does not exist, skipping...")
        return audio_files
    
    for speaker_dir in os.listdir(split_dir):
        speaker_path = os.path.join(split_dir, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
            
        for chapter_dir in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_dir)
            if not os.path.isdir(chapter_path):
                continue
            
            # Find .flac files
            for file in os.listdir(chapter_path):
                if file.endswith('.flac'):
                    file_path = os.path.join(chapter_path, file)
                    # Parse filename: speaker-chapter-segment.flac
                    parts = file.replace('.flac', '').split('-')
                    if len(parts) >= 3:
                        speaker_id = parts[0]
                        chapter_id = parts[1]
                        segment_id = parts[2]
                        
                        # Find corresponding .txt file for transcription
                        txt_file = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
                        transcription = None
                        if os.path.exists(txt_file):
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.startswith(f"{speaker_id}-{chapter_id}-{segment_id}"):
                                        transcription = line.split(' ', 1)[1].strip()
                                        break
                        
                        # Get audio duration
                        try:
                            info = sf.info(file_path)
                            duration = info.duration
                        except:
                            duration = None
                        
                        # Get speaker gender from SPEAKERS.TXT if available
                        gender = None
                        
                        audio_files.append({
                            'audio_file': file_path,
                            'speaker_id': speaker_id,
                            'chapter_id': chapter_id,
                            'segment_id': segment_id,
                            'transcription': transcription or '',
                            'duration': duration,
                            'gender': gender
                        })
    
    return audio_files


def load_speaker_info(root_dir):
    """Load speaker gender information from SPEAKERS.TXT"""
    speakers_file = os.path.join(root_dir, 'SPEAKERS.TXT')
    speaker_info = {}
    
    if os.path.exists(speakers_file):
        with open(speakers_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith(';'):
                    parts = line.strip().split('|')
                    if len(parts) >= 4:
                        speaker_id = parts[0].strip()
                        gender = parts[1].strip().lower()
                        speaker_info[speaker_id] = gender
    
    return speaker_info


def get_audio_duration(audio_path):
    """Get audio duration in seconds"""
    try:
        info = sf.info(audio_path)
        return info.duration
    except:
        try:
            waveform, sr = torchaudio.load(audio_path)
            return waveform.shape[1] / sr
        except:
            return None


def mix_audio_partially_overlapped(wav_paths, delays, sample_rate=16000, output_path=None):
    """
    Mix multiple audio files with partial overlap
    Args:
        wav_paths: List of audio file paths
        delays: List of delays in seconds for each audio
    Returns:
        mixed_waveform: Mixed audio waveform
        duration: Total duration
    """
    waveforms = []
    max_samples = 0
    
    # Load all waveforms
    for wav_path, delay in zip(wav_paths, delays):
        try:
            waveform, sr = torchaudio.load(wav_path)
            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveforms.append((waveform.squeeze(0), delay))
            max_samples = max(max_samples, waveform.shape[1] + int(delay * sample_rate))
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return None, None
    
    if len(waveforms) == 0:
        return None, None
    
    # Create mixed waveform
    mixed = torch.zeros(max_samples)
    
    for waveform, delay in waveforms:
        delay_samples = int(delay * sample_rate)
        end_samples = delay_samples + waveform.shape[0]
        if end_samples > max_samples:
            end_samples = max_samples
            waveform = waveform[:end_samples - delay_samples]
        
        mixed[delay_samples:end_samples] += waveform
    
    # Normalize to prevent clipping
    max_val = torch.abs(mixed).max()
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, mixed.numpy(), sample_rate)
    
    return mixed, max_samples / sample_rate


def generate_speaker_profiles(samples_by_speaker, num_profiles=8, num_utterances_per_profile=2):
    """
    Generate speaker profiles for SA-ASR
    Each profile contains multiple utterances from the same speaker
    """
    profiles = {}
    
    for speaker_id, samples in samples_by_speaker.items():
        if len(samples) < num_utterances_per_profile:
            continue
        
        # Randomly select utterances for profiles
        selected_samples = random.sample(samples, min(num_profiles * num_utterances_per_profile, len(samples)))
        
        profiles[speaker_id] = []
        for i in range(num_profiles):
            start_idx = i * num_utterances_per_profile
            end_idx = start_idx + num_utterances_per_profile
            if end_idx <= len(selected_samples):
                profile_utts = [s['audio_file'] for s in selected_samples[start_idx:end_idx]]
                profiles[speaker_id].append(profile_utts)
    
    return profiles


def create_1mix_dataset(samples, output_dir, split_name, speaker_info):
    """Create 1-speaker dataset (single speaker, no mixing)"""
    print(f"Creating 1-speaker dataset for {split_name}...")
    
    output_subdir = os.path.join(output_dir, f"{split_name}-1mix")
    os.makedirs(output_subdir, exist_ok=True)
    
    jsonl_path = os.path.join(output_dir, f"{split_name}-1mix.jsonl")
    jsonl_file = open(jsonl_path, 'w', encoding='utf-8')
    
    samples_by_speaker = defaultdict(list)
    for sample in samples:
        samples_by_speaker[sample['speaker_id']].append(sample)
    
    # Generate speaker profiles
    profiles = generate_speaker_profiles(samples_by_speaker)
    
    count = 0
    for sample in tqdm(samples, desc="Processing 1mix"):
        speaker_id = sample['speaker_id']
        
        # Copy audio file
        rel_path = os.path.relpath(sample['audio_file'], output_dir)
        output_wav = os.path.join(output_subdir, f"{split_name}-1mix-{count:04d}.wav")
        
        # Load and save as wav
        try:
            waveform, sr = torchaudio.load(sample['audio_file'])
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            os.makedirs(os.path.dirname(output_wav), exist_ok=True)
            sf.write(output_wav, waveform.squeeze(0).numpy(), 16000)
        except Exception as e:
            print(f"Error processing {sample['audio_file']}: {e}")
            continue
        
        # Get speaker profile
        speaker_profile = []
        speaker_profile_index = None
        if speaker_id in profiles and len(profiles[speaker_id]) > 0:
            speaker_profile = profiles[speaker_id]
            speaker_profile_index = 0  # Only one speaker in 1mix
        
        # Create JSON entry
        entry = {
            "id": f"{split_name}-1mix/{split_name}-1mix-{count:04d}",
            "mixed_wav": os.path.relpath(output_wav, output_dir),
            "texts": [sample['transcription']],
            "speaker_profile": speaker_profile,
            "speaker_profile_index": [speaker_profile_index] if speaker_profile_index is not None else None,
            "wavs": [os.path.relpath(sample['audio_file'], output_dir)],
            "delays": [0.0],
            "speakers": [speaker_id],
            "durations": [sample['duration'] or get_audio_duration(sample['audio_file'])],
            "genders": [speaker_info.get(speaker_id, '')]
        }
        
        jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        count += 1
    
    jsonl_file.close()
    print(f"Created {count} 1-speaker samples: {jsonl_path}")


def create_nmix_dataset(samples, output_dir, split_name, n_speakers, speaker_info, 
                       min_overlap_ratio=0.1, max_overlap_ratio=0.5):
    """
    Create n-speaker mixed dataset with partial overlap
    Args:
        n_speakers: Number of speakers (2 or 3)
        min_overlap_ratio: Minimum overlap ratio (0.0-1.0)
        max_overlap_ratio: Maximum overlap ratio (0.0-1.0)
    """
    print(f"Creating {n_speakers}-speaker mixed dataset for {split_name}...")
    
    output_subdir = os.path.join(output_dir, f"{split_name}-{n_speakers}mix")
    os.makedirs(output_subdir, exist_ok=True)
    
    jsonl_path = os.path.join(output_dir, f"{split_name}-{n_speakers}mix.jsonl")
    jsonl_file = open(jsonl_path, 'w', encoding='utf-8')
    
    # Group samples by speaker
    samples_by_speaker = defaultdict(list)
    for sample in samples:
        samples_by_speaker[sample['speaker_id']].append(sample)
    
    speaker_ids = list(samples_by_speaker.keys())
    
    # Generate speaker profiles
    profiles = generate_speaker_profiles(samples_by_speaker)
    
    # Use each sample exactly n_speakers times (as per LibriSpeechMix design)
    # Track usage count for each sample
    sample_usage_count = defaultdict(int)
    count = 0
    
    for base_sample in tqdm(samples, desc=f"Processing {n_speakers}mix"):
        # Skip if this sample has been used n_speakers times already
        if sample_usage_count[base_sample['audio_file']] >= n_speakers:
            continue
        
        # Select n_speakers different speakers
        selected_speakers = [base_sample['speaker_id']]
        available_speakers = [s for s in speaker_ids if s != base_sample['speaker_id']]
        
        if len(available_speakers) < n_speakers - 1:
            continue
        
        selected_speakers.extend(random.sample(available_speakers, n_speakers - 1))
        
        # Select one sample from each speaker
        selected_samples = [base_sample]
        for spk_id in selected_speakers[1:]:
            if len(samples_by_speaker[spk_id]) > 0:
                # Select a sample that hasn't been used n_speakers times yet
                available = [s for s in samples_by_speaker[spk_id] 
                           if sample_usage_count[s['audio_file']] < n_speakers]
                if len(available) == 0:
                    # If all samples are used, allow reuse
                    available = samples_by_speaker[spk_id]
                selected_samples.append(random.choice(available))
            else:
                break
        
        if len(selected_samples) < n_speakers:
            continue
        
        # Get durations
        durations = []
        for sample in selected_samples:
            dur = sample['duration'] or get_audio_duration(sample['audio_file'])
            if dur is None:
                break
            durations.append(dur)
        
        if len(durations) < n_speakers:
            continue
        
        # Calculate delays for partial overlap
        # First speaker starts at 0
        delays = [0.0]
        
        # Calculate overlap for subsequent speakers
        for i in range(1, n_speakers):
            # Overlap ratio between previous and current speaker
            overlap_ratio = random.uniform(min_overlap_ratio, max_overlap_ratio)
            
            # Previous speaker's end time
            prev_end = delays[i-1] + durations[i-1]
            
            # Current speaker should start before previous ends
            overlap_duration = durations[i-1] * overlap_ratio
            current_start = prev_end - overlap_duration
            
            # Ensure current speaker doesn't start before 0
            current_start = max(0.0, current_start)
            delays.append(current_start)
        
        # Mix audio
        wav_paths = [s['audio_file'] for s in selected_samples]
        output_wav = os.path.join(output_subdir, f"{split_name}-{n_speakers}mix-{count:04d}.wav")
        
        mixed_waveform, total_duration = mix_audio_partially_overlapped(
            wav_paths, delays, sample_rate=16000, output_path=output_wav
        )
        
        if mixed_waveform is None:
            continue
        
        # Increment usage count for each sample
        for sample in selected_samples:
            sample_usage_count[sample['audio_file']] += 1
        
        # Get speaker profiles
        speaker_profile = []
        speaker_profile_index = []
        for i, sample in enumerate(selected_samples):
            spk_id = sample['speaker_id']
            if spk_id in profiles and len(profiles[spk_id]) > 0:
                if i == 0:
                    speaker_profile = profiles[spk_id]
                speaker_profile_index.append(i)
        
        # Create JSON entry
        entry = {
            "id": f"{split_name}-{n_speakers}mix/{split_name}-{n_speakers}mix-{count:04d}",
            "mixed_wav": os.path.relpath(output_wav, output_dir),
            "texts": [s['transcription'] for s in selected_samples],
            "speaker_profile": speaker_profile,
            "speaker_profile_index": speaker_profile_index if speaker_profile else None,
            "wavs": [os.path.relpath(wav, output_dir) for wav in wav_paths],
            "delays": delays,
            "speakers": [s['speaker_id'] for s in selected_samples],
            "durations": durations,
            "genders": [speaker_info.get(s['speaker_id'], '') for s in selected_samples]
        }
        
        jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        count += 1
    
    jsonl_file.close()
    print(f"Created {count} {n_speakers}-speaker mixed samples: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare LibriSpeechMix evaluation dataset')
    parser.add_argument('--root_dir', type=str,
                       default='/data/work/SimonFang/datasets/english/LibriSpeech',
                       help='Root directory of LibriSpeech dataset')
    parser.add_argument('--output_dir', type=str, default='./data/librispeechmix',
                       help='Output directory for LibriSpeechMix dataset')
    parser.add_argument('--splits', type=str, nargs='+', default=['dev-clean', 'test-clean'],
                       help='Splits to process')
    parser.add_argument('--min_overlap', type=float, default=0.1,
                       help='Minimum overlap ratio for mixed speech')
    parser.add_argument('--max_overlap', type=float, default=0.5,
                       help='Maximum overlap ratio for mixed speech')
    
    args = parser.parse_args()
    
    # Load speaker information
    speaker_info = load_speaker_info(args.root_dir)
    print(f"Loaded gender info for {len(speaker_info)} speakers")
    
    # Process each split
    for split_name in args.splits:
        print("=" * 60)
        print(f"Processing {split_name}...")
        print("=" * 60)
        
        # Find all audio files
        samples = find_audio_files(args.root_dir, split_name)
        print(f"Found {len(samples)} audio files")
        
        if len(samples) == 0:
            print(f"Warning: No samples found for {split_name}, skipping...")
            continue
        
        # Create 1-speaker dataset
        create_1mix_dataset(samples, args.output_dir, split_name, speaker_info)
        
        # Create 2-speaker mixed dataset
        create_nmix_dataset(samples, args.output_dir, split_name, 2, speaker_info,
                          args.min_overlap, args.max_overlap)
        
        # Create 3-speaker mixed dataset
        create_nmix_dataset(samples, args.output_dir, split_name, 3, speaker_info,
                          args.min_overlap, args.max_overlap)
    
    print("=" * 60)
    print("LibriSpeechMix dataset generation completed!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated files:")
    for split_name in args.splits:
        print(f"  - {split_name}-1mix.jsonl")
        print(f"  - {split_name}-2mix.jsonl")
        print(f"  - {split_name}-3mix.jsonl")


if __name__ == '__main__':
    main()

