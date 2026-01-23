#!/usr/bin/env python3
"""
Prepare manifest.json files for LibriSpeech dataset
Supports:
- Single speaker manifests (train, dev, test)
- Multi-speaker mixed manifests (on-the-fly training)

Copyright (c) 2026 Simon Fang
Email: fangshuming519@gmail.com

MIT License
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random


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
                        speaker_id = int(parts[0])
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
                        
                        audio_files.append({
                            'audio_file': file_path,
                            'speaker_id': int(speaker_id),  # Ensure integer
                            'chapter_id': chapter_id,
                            'segment_id': segment_id,
                            'transcription': transcription or '',
                            'duration': None  # Will be filled if needed
                        })
    
    return audio_files


def create_single_speaker_manifest(root_dir, split_name, output_file, min_duration=1.0, max_duration=30.0, max_samples=None):
    """Create manifest for single speaker data"""
    print(f"Creating manifest for {split_name}...")
    audio_files = find_audio_files(root_dir, split_name)
    
    # Limit samples if specified (for testing)
    if max_samples is not None and max_samples > 0:
        if len(audio_files) > max_samples:
            print(f"Limiting to {max_samples} samples (found {len(audio_files)})")
            audio_files = audio_files[:max_samples]
    
    # Filter by duration if audio duration info is available
    # For now, we'll include all files and filter during loading
    
    manifest = {
        'root_dir': root_dir,
        'split': split_name,
        'samples': audio_files
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Created manifest with {len(audio_files)} samples: {output_file}")
    return manifest


def create_mix_manifest(root_dir, train_splits, output_file, num_samples=10000, 
                        max_speakers=2, overlap_ratio=0.5):  # Paper uses 2 speakers, 0.5 overlap
    """
    Create mixing configurations for multi-speaker training data.
    
    IMPORTANT: This creates MIXING CONFIGURATIONS (metadata), NOT pre-mixed audio files.
    The actual audio mixing happens on-the-fly during training in LibriSpeechMixDataset.
    
    NOTE: Pre-generating configurations is OPTIONAL. If num_samples=0 or None,
    the dataset will perform truly on-the-fly random mixing (recommended for reproduction).
    Pre-generated configs are mainly for reproducibility and consistency across runs.
    
    Args:
        num_samples: Number of mixing configurations to pre-generate.
                     If 0 or None, no configs are generated (truly on-the-fly).
                     These configurations define which speakers/samples to mix,
                     but the actual mixing is done dynamically during training.
    """
    print(f"Creating mix manifest from {train_splits}...")
    
    # Collect all audio files from training splits
    all_samples = []
    for split in train_splits:
        samples = find_audio_files(root_dir, split)
        all_samples.extend(samples)
    
    if len(all_samples) == 0:
        print("Warning: No samples found for mixing!")
        return None
    
    # Group by speaker for easier sampling
    speaker_groups = {}
    for sample in all_samples:
        spk_id = sample['speaker_id']
        if spk_id not in speaker_groups:
            speaker_groups[spk_id] = []
        speaker_groups[spk_id].append(sample)
    
    # Create mix configurations (metadata for on-the-fly mixing)
    mix_configs = []
    speakers_list = list(speaker_groups.keys())
    
    print(f"Generating {num_samples} mix configurations...")
    for _ in tqdm(range(num_samples)):
        # Randomly select number of speakers (2 to max_speakers)
        num_spks = random.randint(2, max_speakers)
        
        # Randomly select speakers
        selected_speakers = random.sample(speakers_list, min(num_spks, len(speakers_list)))
        
        # For each speaker, randomly select a sample
        mix_config = {
            'speakers': [],
            'overlap_ratio': overlap_ratio
        }
        
        for spk_id in selected_speakers:
            if len(speaker_groups[spk_id]) > 0:
                sample = random.choice(speaker_groups[spk_id])
                mix_config['speakers'].append({
                    'speaker_id': spk_id,
                    'sample_idx': all_samples.index(sample)  # Reference to original sample
                })
        
        if len(mix_config['speakers']) >= 2:
            mix_configs.append(mix_config)
    
    manifest = {
        'root_dir': root_dir,
        'type': 'mix',
        'base_samples': all_samples,  # All available samples
        'mix_configs': mix_configs,   # Mixing configurations
        'overlap_ratio': overlap_ratio,
        'max_speakers': max_speakers
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Created mix manifest with {len(mix_configs)} mix configurations: {output_file}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Prepare LibriSpeech manifests')
    parser.add_argument('--root_dir', type=str, 
                       default='/data/work/SimonFang/datasets/english/LibriSpeech',
                       help='Root directory of LibriSpeech dataset')
    parser.add_argument('--output_dir', type=str, default='./data/manifests',
                       help='Output directory for manifest files')
    parser.add_argument('--mix_samples', type=int, default=0,
                       help='Number of mix configurations to pre-generate. '
                            'If 0, truly on-the-fly random mixing is used (recommended). '
                            'Pre-generated configs are for reproducibility only.')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create single speaker manifests
    print("=" * 60)
    print("Creating single speaker manifests...")
    print("=" * 60)
    
    train_manifest = create_single_speaker_manifest(
        args.root_dir, 'train-clean-100',
        os.path.join(output_dir, 'train_manifest.json')
    )
    
    # Also include train-clean-360 and train-other-500 if available
    train_360 = find_audio_files(args.root_dir, 'train-clean-360')
    train_500 = find_audio_files(args.root_dir, 'train-other-500')
    
    if train_360:
        print(f"Found {len(train_360)} samples in train-clean-360")
        train_manifest['samples'].extend(train_360)
    
    if train_500:
        print(f"Found {len(train_500)} samples in train-other-500")
        train_manifest['samples'].extend(train_500)
    
    # Save updated train manifest
    with open(os.path.join(output_dir, 'train_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(train_manifest, f, indent=2, ensure_ascii=False)
    print(f"Updated train manifest with {len(train_manifest['samples'])} total samples")
    
    create_single_speaker_manifest(
        args.root_dir, 'dev-clean',
        os.path.join(output_dir, 'dev_manifest.json')
    )
    
    create_single_speaker_manifest(
        args.root_dir, 'test-clean',
        os.path.join(output_dir, 'test_manifest.json')
    )
    
    # Create mix configurations for on-the-fly training (OPTIONAL)
    # Note: Pre-generating configs is OPTIONAL. If mix_samples=0, truly on-the-fly mixing is used.
    # Pre-generated configs are mainly for reproducibility, not required by the paper.
    print("=" * 60)
    if args.mix_samples > 0:
        print("Creating mix configurations for on-the-fly training...")
        print("Note: These are mixing configurations (metadata), not pre-mixed audio files.")
        print("Actual mixing happens dynamically during training.")
        print("Pre-generated configs are for reproducibility, not required by SA-SOT paper.")
    else:
        print("Skipping mix configuration generation (truly on-the-fly mode).")
        print("Mixing will be completely random during training (recommended for reproduction).")
    print("=" * 60)
    
    if args.mix_samples > 0:
        create_mix_manifest(
            args.root_dir,
            ['train-clean-100', 'train-clean-360', 'train-other-500'],
            os.path.join(output_dir, 'mix_train_manifest.json'),
            num_samples=args.mix_samples,  # Number of mix configurations to pre-generate
            max_speakers=2,  # Paper only supports 2 speakers
            overlap_ratio=0.5  # Paper uses 0.5 overlap ratio
        )
    else:
        # Create a minimal manifest with just base samples for truly on-the-fly mixing
        all_samples = []
        for split in ['train-clean-100', 'train-clean-360', 'train-other-500']:
            samples = find_audio_files(args.root_dir, split)
            all_samples.extend(samples)
        
        manifest = {
            'root_dir': args.root_dir,
            'type': 'mix_on_the_fly',
            'base_samples': all_samples,
            'mix_configs': []  # Empty - will be generated on-the-fly
        }
        
        output_file = os.path.join(output_dir, 'mix_train_manifest.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Created on-the-fly mix manifest with {len(all_samples)} base samples: {output_file}")
    
    print("=" * 60)
    print("All manifests created successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

