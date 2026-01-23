"""
LibriSpeech dataset loaders for SA-SOT training
Supports:
- Single speaker datasets
- Multi-speaker on-the-fly mixing

Copyright (c) 2026 Simon Fang
Email: fangshuming519@gmail.com

MIT License
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
import torchaudio
from collections import defaultdict


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset for single speaker training"""
    
    def __init__(self, manifest_path, tokenizer=None, sample_rate=16000, 
                 n_mels=80, hop_length=160, win_length=400, n_fft=512,
                 min_duration=1.0, max_duration=30.0, word_alignments=None):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.word_alignments = word_alignments  # Dict mapping utterance_id -> word alignments
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        self.samples = self.manifest['samples']
        
        # Filter by duration if needed
        self.samples = [s for s in self.samples if self._check_duration(s)]
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def _check_duration(self, sample):
        """Check if sample duration is within range"""
        # For now, accept all samples
        # Duration filtering can be done after loading audio
        return True
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Check duration
            duration = waveform.shape[1] / self.sample_rate
            if duration < self.min_duration or duration > self.max_duration:
                return None, None
            
            return waveform, duration
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None
    
    def _audio_to_feats(self, waveform):
        """Convert waveform to mel spectrogram features"""
        # waveform: (1, T)
        mel = self.mel_transform(waveform)  # (1, n_mels, T')
        mel = mel.squeeze(0).transpose(0, 1)  # (T', n_mels)
        return mel
    
    def _text_to_tokens(self, text):
        """Convert text to token IDs"""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        else:
            # Simple character-level tokenization as fallback
            # In practice, you should use a proper tokenizer
            return [ord(c) % 1000 for c in text[:100]]  # Simple fallback
    
    def __len__(self):
        return len(self.samples)
    
    def _get_utterance_id(self, sample):
        """Extract utterance ID from sample (for MFA alignment lookup)"""
        # Format: speaker-chapter-segment
        speaker_id = sample.get('speaker_id', '')
        chapter_id = sample.get('chapter_id', '')
        segment_id = sample.get('segment_id', '')
        if speaker_id and chapter_id and segment_id:
            return f"{speaker_id}-{chapter_id}-{segment_id}"
        return None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        waveform, duration = self._load_audio(sample['audio_file'])
        if waveform is None:
            # Return a dummy sample if loading fails
            return self._get_dummy_sample()
        
        # Extract features
        feats = self._audio_to_feats(waveform)  # (T, n_mels)
        feat_len = feats.shape[0]
        
        # Tokenize text
        text = sample.get('transcription', '')
        if text:
            tokens = self._text_to_tokens(text)
        else:
            # Generate dummy tokens if no transcription
            tokens = [1] * max(5, feat_len // 4)
        
        # Get word emission times from MFA alignment if available
        word_emission_times = None
        if self.word_alignments is not None:
            utt_id = self._get_utterance_id(sample)
            if utt_id and utt_id in self.word_alignments:
                word_emission_times = self.word_alignments[utt_id]
        
        return {
            'feats': feats,
            'feat_lens': feat_len,
            'tokens': tokens,
            'speaker_id': sample['speaker_id'],
            'audio_file': sample['audio_file'],
            'word_emission_times': word_emission_times  # List of {word, start, end, duration}
        }
    
    def _get_dummy_sample(self):
        """Return a dummy sample for failed loads"""
        feats = torch.randn(100, self.n_mels)
        return {
            'feats': feats,
            'feat_lens': 100,
            'tokens': [1] * 25,
            'speaker_id': 0,
            'audio_file': ''
        }


class LibriSpeechMixDataset(Dataset):
    """
    LibriSpeech dataset with on-the-fly multi-speaker mixing.
    
    If mix_configs are provided in manifest, uses pre-generated configurations.
    Otherwise, performs truly on-the-fly random mixing during training.
    """
    
    def __init__(self, manifest_path, tokenizer=None, sample_rate=16000,
                 n_mels=80, hop_length=160, win_length=400, n_fft=512,
                 min_duration=1.0, max_duration=30.0, overlap_ratio=0.5,  # Paper uses 0.5
                 cc_id=3, max_speakers=2, truly_on_the_fly=False):  # Paper only supports 2 speakers
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.overlap_ratio = overlap_ratio
        self.cc_id = cc_id
        self.max_speakers = max_speakers
        self.truly_on_the_fly = truly_on_the_fly
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        self.base_samples = self.manifest.get('base_samples', self.manifest.get('samples', []))
        
        if truly_on_the_fly:
            # Truly on-the-fly: no pre-generated configs, mix randomly each time
            self.mix_configs = None
            # Group samples by speaker for efficient random sampling
            self.samples_by_speaker = defaultdict(list)
            for sample in self.base_samples:
                self.samples_by_speaker[sample['speaker_id']].append(sample)
            self.speaker_ids = list(self.samples_by_speaker.keys())
        else:
            # Use pre-generated mix configurations
            self.mix_configs = self.manifest.get('mix_configs', [])
            self.samples_by_speaker = None
            self.speaker_ids = None
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform.squeeze(0)  # (T,)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def _audio_to_feats(self, waveform):
        """Convert waveform to mel spectrogram features"""
        waveform = waveform.unsqueeze(0)  # (1, T)
        mel = self.mel_transform(waveform)  # (1, n_mels, T')
        mel = mel.squeeze(0).transpose(0, 1)  # (T', n_mels)
        return mel
    
    def _text_to_tokens(self, text):
        """Convert text to token IDs"""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        else:
            return [ord(c) % 1000 for c in text[:100]]
    
    def _mix_audio(self, waveforms, overlap_ratio=0.5):  # Paper uses 0.5
        """
        Mix multiple waveforms with overlap
        Returns: mixed waveform, alignment info for t-SOT label generation
        """
        if len(waveforms) == 1:
            return waveforms[0], [(0, len(waveforms[0]))]
        
        # Calculate overlap length
        max_len = max(len(w) for w in waveforms)
        overlap_samples = int(max_len * overlap_ratio)
        
        # Mix waveforms with overlap
        # Calculate total length accounting for overlaps
        total_len = sum(len(w) for w in waveforms) - overlap_samples * (len(waveforms) - 1)
        mixed = torch.zeros(total_len)
        
        # Track alignment for each speaker
        alignments = []
        current_pos = 0
        
        for i, wav in enumerate(waveforms):
            wav_len = len(wav)
            start = current_pos
            
            if i == 0:
                # First speaker: place at the beginning
                end = start + wav_len
                if end > len(mixed):
                    # Extend if needed
                    mixed = torch.cat([mixed, torch.zeros(end - len(mixed))])
                mixed[start:end] = wav
            else:
                # Subsequent speakers: overlap with previous
                overlap_start = max(0, start - overlap_samples)
                overlap_len = start - overlap_start
                actual_overlap = min(overlap_len, wav_len)
                
                # Mix overlapping region
                if actual_overlap > 0:
                    mixed[overlap_start:overlap_start+actual_overlap] += wav[:actual_overlap]
                
                # Add non-overlapping part
                # Ensure actual_overlap is strictly less than wav_len to get non-empty slice
                if actual_overlap < wav_len and actual_overlap >= 0:
                    non_overlap_len = wav_len - actual_overlap
                    if non_overlap_len > 0:
                        non_overlap_start = start
                        non_overlap_end = non_overlap_start + non_overlap_len
                        if non_overlap_end > len(mixed):
                            # Extend if needed
                            mixed = torch.cat([mixed, torch.zeros(non_overlap_end - len(mixed))])
                        # Get the non-overlapping part using explicit slice
                        # Ensure we don't exceed wav length
                        slice_end = min(actual_overlap + non_overlap_len, wav_len)
                        wav_non_overlap = wav[actual_overlap:slice_end]
                        # Double-check the slice is not empty and has correct length
                        actual_slice_len = len(wav_non_overlap)
                        if actual_slice_len > 0 and actual_slice_len == (slice_end - actual_overlap):
                            # Adjust end position to match actual slice length
                            actual_end = non_overlap_start + actual_slice_len
                            if actual_end > len(mixed):
                                mixed = torch.cat([mixed, torch.zeros(actual_end - len(mixed))])
                            # Use index assignment
                            mixed[non_overlap_start:actual_end] = wav_non_overlap.clone()
                            end = actual_end
                        else:
                            # Slice is empty or wrong size - skip
                            end = start
                    else:
                        end = start
                else:
                    # All audio is in overlap region (actual_overlap >= wav_len)
                    end = start
            alignments.append((start, end))
            current_pos = end - overlap_samples
        
        return mixed, alignments
    
    def _build_t_sot_label(self, tokens_list, speaker_ids, alignments):
        """
        Build t-SOT (tokenized Serialized Output Training) label
        Format: [spk1_tokens] <cc> [spk2_tokens] <cc> [spk3_tokens] ...
        """
        t_sot_tokens = []
        t_sot_speaker_ids = []
        
        for i, (tokens, spk_id) in enumerate(zip(tokens_list, speaker_ids)):
            if i > 0:
                # Add change-of-character token
                t_sot_tokens.append(self.cc_id)
                t_sot_speaker_ids.append(-1)  # cc_id doesn't belong to any speaker
            
            # Add speaker's tokens
            t_sot_tokens.extend(tokens)
            t_sot_speaker_ids.extend([spk_id] * len(tokens))
        
        return t_sot_tokens, t_sot_speaker_ids
    
    def __len__(self):
        if self.truly_on_the_fly:
            # For truly on-the-fly, return length of base samples
            # (each can be used multiple times in different mixes)
            return len(self.base_samples)
        else:
            return len(self.mix_configs)
    
    def _generate_random_mix_config(self):
        """Generate a random mixing configuration on-the-fly"""
        # Randomly select number of speakers (2 to max_speakers)
        num_spks = random.randint(2, self.max_speakers)
        
        # Randomly select speakers
        if len(self.speaker_ids) < num_spks:
            num_spks = len(self.speaker_ids)
        
        selected_speakers = random.sample(self.speaker_ids, num_spks)
        
        # For each speaker, randomly select a sample
        mix_config = {
            'speakers': [],
            'overlap_ratio': self.overlap_ratio
        }
        
        for spk_id in selected_speakers:
            if len(self.samples_by_speaker[spk_id]) > 0:
                sample = random.choice(self.samples_by_speaker[spk_id])
                mix_config['speakers'].append({
                    'speaker_id': spk_id,
                    'sample': sample  # Store sample directly
                })
        
        return mix_config
    
    def __getitem__(self, idx):
        if self.truly_on_the_fly:
            # Generate random mix configuration on-the-fly
            mix_config = self._generate_random_mix_config()
        else:
            # Use pre-generated mix configuration
            mix_config = self.mix_configs[idx]
        
        # Load audio and text for each speaker
        waveforms = []
        tokens_list = []
        speaker_ids = []
        
        for spk_info in mix_config['speakers']:
            # Handle both pre-generated configs (with sample_idx) and on-the-fly (with sample)
            if 'sample' in spk_info:
                sample = spk_info['sample']
            else:
                sample_idx = spk_info['sample_idx']
                sample = self.base_samples[sample_idx]
            
            # Load audio
            waveform = self._load_audio(sample['audio_file'])
            if waveform is None:
                # Skip this speaker if loading fails
                continue
            
            waveforms.append(waveform)
            speaker_ids.append(spk_info['speaker_id'])
            
            # Tokenize text
            text = sample.get('transcription', '')
            if text:
                tokens = self._text_to_tokens(text)
            else:
                tokens = [1] * max(5, len(waveform) // (self.sample_rate // 4))
            tokens_list.append(tokens)
        
        if len(waveforms) < 2:
            # Fallback to single speaker
            return self._get_single_speaker_sample(waveforms[0] if waveforms else None, 
                                                   tokens_list[0] if tokens_list else [1]*10,
                                                   speaker_ids[0] if speaker_ids else 0)
        
        # Mix audio
        overlap_ratio = mix_config.get('overlap_ratio', self.overlap_ratio)
        mixed_waveform, alignments = self._mix_audio(waveforms, overlap_ratio)
        
        # Extract features
        feats = self._audio_to_feats(mixed_waveform)  # (T, n_mels)
        feat_len = feats.shape[0]
        
        # Build t-SOT label
        t_sot_tokens, t_sot_speaker_ids = self._build_t_sot_label(
            tokens_list, speaker_ids, alignments
        )
        
        return {
            'feats': feats,
            'feat_lens': feat_len,
            'tokens': t_sot_tokens,
            'speaker_ids': t_sot_speaker_ids,  # (N,) per-token speaker IDs
            'audio_file': f"mix_{idx}"
        }
    
    def _get_single_speaker_sample(self, waveform, tokens, speaker_id):
        """Fallback for single speaker sample"""
        if waveform is None:
            waveform = torch.randn(16000)  # 1 second of audio
        
        feats = self._audio_to_feats(waveform)
        return {
            'feats': feats,
            'feat_lens': feats.shape[0],
            'tokens': tokens,
            'speaker_ids': [speaker_id] * len(tokens),
            'audio_file': ''
        }
