#!/usr/bin/env python3
"""
SA-SOT Model Training Script

Copyright (c) 2026 Simon Fang
Email: fangshuming519@gmail.com

MIT License
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse

from model.sasot_model import SASOTModel
from loss.asr_loss import masked_ce_loss
from loss.speaker_loss import SpeakerAMSoftmax
from data.librispeech import LibriSpeechDataset, LibriSpeechMixDataset
from data.tokenizer import BPETokenizer
from data.load_alignments import load_word_alignments


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """Batch collation function - process t-SOT format data"""
    batch_size = len(batch)
    
    # Get maximum lengths
    max_feat_len = max(item['feat_lens'] for item in batch)
    max_target_len = max(len(item['tokens']) for item in batch)
    
    # Initialize batch tensors
    feats = torch.zeros(batch_size, max_feat_len, 80)
    feat_lens = torch.zeros(batch_size, dtype=torch.long)
    target_tokens = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    target_lens = torch.zeros(batch_size, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, max_target_len, dtype=torch.long)  # (B, N)
    
    # Pad data
    for i, item in enumerate(batch):
        feat_len = item['feat_lens']
        target_len = len(item['tokens'])
        
        feats[i, :feat_len] = item['feats']
        feat_lens[i] = feat_len
        target_tokens[i, :target_len] = torch.tensor(item['tokens'], dtype=torch.long)
        target_lens[i] = target_len
        
        # speaker_ids: (N,) -> (max_target_len,)
        spk_ids = item.get('speaker_ids', [item.get('speaker_id', 0)] * target_len)
        if isinstance(spk_ids, list):
            spk_ids = torch.tensor(spk_ids[:target_len], dtype=torch.long)
        else:
            spk_ids = torch.tensor([spk_ids] * target_len, dtype=torch.long)
        speaker_ids[i, :target_len] = spk_ids
    
    return {
        'feats': feats,
        'feat_lens': feat_lens,
        'target_tokens': target_tokens,
        'target_lens': target_lens,
        'speaker_ids': speaker_ids  # (B, N) speaker id for each token
    }


def create_loss_mask(target_lens, max_len):
    """Create loss mask"""
    batch_size = target_lens.size(0)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(target_lens):
        mask[i, :length] = True
    return mask


def train_epoch(model, dataloader, optimizer, device, vocab_size, num_speakers, speaker_loss_fn, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_asr_loss = 0.0
    total_spk_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        feats = batch['feats'].to(device)
        feat_lens = batch['feat_lens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        target_lens = batch['target_lens'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        
        batch_size = feats.size(0)
        
        # Prepare input
        # prev_tokens: use shifted target tokens for teacher forcing during training
        prev_tokens = torch.zeros_like(target_tokens)
        prev_tokens[:, 1:] = target_tokens[:, :-1]
        prev_tokens[:, 0] = 0  # <sos> token
        
        # Forward pass
        output = model(
            feats,
            feat_lens,
            prev_tokens,
            target_len=target_lens,
            speaker_ids=speaker_ids,
            return_intermediate=True
        )
        
        # Calculate ASR loss
        # During training with SAT, output is a dict {(b, spk): logits}
        # During inference, output is a tensor (B, N, V)
        asr_logits = output['asr_logits']
        
        if isinstance(asr_logits, dict):
            # SAT training mode: calculate masked loss for each speaker
            asr_losses = []
            # Get token_lens from output to ensure length matching
            token_lens = output.get('token_lens', target_lens)  # (B,)
            
            for (b, spk), logits in asr_logits.items():
                # Get actual length: use minimum of target_len and token_lens
                actual_target_len = target_lens[b].item()
                actual_token_len = token_lens[b].item()
                actual_len = min(actual_target_len, actual_token_len)
                actual_len = max(1, actual_len)  # Ensure at least 1
                
                # Truncate all tensors to actual_len to ensure length matching
                batch_target = target_tokens[b:b+1, :actual_len]  # (1, actual_len)
                batch_logits = logits  # (1, actual_len, V) - already truncated in model
                batch_speaker_ids = speaker_ids[b:b+1, :actual_len]  # (1, actual_len)
                
                # mask: True for tokens belonging to current speaker (and not cc_id)
                # Handle DDP wrapped model
                actual_model = model.module if hasattr(model, 'module') else model
                mask_id = actual_model.mask_id
                cc_id = actual_model.cc_id
                loss_mask = (batch_speaker_ids == spk) & (batch_target != cc_id)
                
                if loss_mask.any():
                    loss = masked_ce_loss(batch_logits, batch_target, loss_mask)
                    asr_losses.append(loss)
            
            if len(asr_losses) > 0:
                asr_loss = sum(asr_losses) / len(asr_losses)
            else:
                asr_loss = torch.tensor(0.0, device=device)
        else:
            # Non-SAT mode (inference or no speaker_ids)
            max_target_len = target_tokens.size(1)
            loss_mask = create_loss_mask(target_lens, max_target_len).to(device)
            asr_loss = masked_ce_loss(asr_logits, target_tokens, loss_mask)
        
        # Calculate Speaker loss
        token_spk = output['token_spk']  # (B, N, D)
        # speaker_ids is already in (B, N) format, each token corresponds to a speaker id
        # Filter out cc_id positions and invalid speaker IDs
        num_speakers = config['model']['num_speakers']
        masked_speaker_ids = speaker_ids.clone()
        
        # Create mask for valid positions (not cc_id and valid speaker ID range)
        # Handle DDP wrapped model
        actual_model = model.module if hasattr(model, 'module') else model
        invalid_mask = (speaker_ids < 0) | (speaker_ids >= num_speakers) | (target_tokens == actual_model.cc_id)
        masked_speaker_ids[invalid_mask] = 0  # Set to first speaker as placeholder
        
        # Only compute loss on valid tokens (not cc_id and valid speaker IDs)
        valid_mask = ~invalid_mask
        if valid_mask.sum() > 0:
            # Flatten and filter
            valid_token_spk = token_spk[valid_mask]  # (valid_tokens, D)
            valid_speaker_ids = masked_speaker_ids[valid_mask]  # (valid_tokens,)
            
            # SpeakerAMSoftmax expects (B*N, D) and (B*N,)
            spk_loss = speaker_loss_fn(valid_token_spk.unsqueeze(0), valid_speaker_ids.unsqueeze(0))
        else:
            spk_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        asr_weight = config['training']['asr_loss_weight']
        spk_weight = config['training']['speaker_loss_weight']
        total_loss_batch = asr_weight * asr_loss + spk_weight * spk_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        grad_clip = config['training'].get('gradient_clip', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_asr_loss += asr_loss.item()
        total_spk_loss += spk_loss.item()
        num_batches += 1
        
        # Update progress bar
        # Show weighted total loss and raw component losses
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'asr_loss': f'{asr_loss.item():.4f}',
            'speaker_loss': f'{spk_loss.item():.4f}',
            'w': f'{asr_weight:.1f}+{spk_weight:.1f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_asr_loss = total_asr_loss / num_batches
    avg_spk_loss = total_spk_loss / num_batches
    
    return avg_loss, avg_asr_loss, avg_spk_loss


def main():
    parser = argparse.ArgumentParser(description='Train SA-SOT model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--use_mix', action='store_true', default=True,
                       help='Use on-the-fly mixed training data (default: True)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility (default: 1234)')
    parser.add_argument('--train_manifest', type=str, default=None,
                       help='Path to training manifest (overrides config)')
    parser.add_argument('--exp_dir', type=str, default=None,
                       help='Experiment directory (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--reset', action='store_true', default=False,
                       help='Reset training: remove old checkpoints and start fresh')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.train_manifest:
        config['data']['train_manifest'] = args.train_manifest
    if args.exp_dir:
        config['training']['save_dir'] = args.exp_dir
    
    # Check for distributed training
    use_distributed = False
    local_rank = -1
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        use_distributed = True
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if local_rank == 0:
            print(f"Using distributed training with {torch.distributed.get_world_size()} GPUs")
    else:
        # Use device from config, or default to 'cuda' if available, else 'cpu'
        device_str = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_str)
    
    # Print training info (only on rank 0 for distributed training)
    if not use_distributed or local_rank == 0:
        print("=" * 50)
        print("SA-SOT Model Training")
        print("=" * 50)
        print(f"Config file: {args.config}")
        
        # Show actual physical GPU info if CUDA_VISIBLE_DEVICES is set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and torch.cuda.is_available():
            physical_gpus = cuda_visible.split(',')
            if use_distributed:
                print(f"Physical GPUs: {cuda_visible} (mapped to cuda:0-{len(physical_gpus)-1})")
            else:
                print(f"Physical GPU: {physical_gpus[0]} (mapped to {device})")
        else:
            print(f"Device: {device}")
        
        print(f"Vocab size: {config['model']['vocab_size']}")
        print(f"Number of speakers: {config['model']['num_speakers']}")
        print(f"Batch size: {config['training']['batch_size']}")
        print(f"Number of epochs: {config['training']['num_epochs']}")
        print(f"Use mix data: {args.use_mix}")
        if use_distributed:
            print(f"Distributed training: {torch.distributed.get_world_size()} GPUs")
        print("=" * 50)
    
    # Create save directory
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle reset option: remove old checkpoints
    if args.reset and (not use_distributed or local_rank == 0):
        import glob
        old_checkpoints = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pt'))
        if old_checkpoints:
            print(f"\n⚠️  RESET MODE: Removing {len(old_checkpoints)} old checkpoints...")
            for ckpt in old_checkpoints:
                os.remove(ckpt)
                print(f"  Removed: {os.path.basename(ckpt)}")
            print("  Starting fresh training with new model architecture.\n")
        args.resume = None  # Don't resume if resetting
    
    # Create dataset and data loader
    data_config = config['data']
    
    # Load BPE tokenizer if specified
    tokenizer = None
    tokenizer_config = config.get('tokenizer', {})
    tokenizer_path = tokenizer_config.get('model_path')
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"Loading BPE tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer(tokenizer_path)
        # Update config with tokenizer IDs
        if tokenizer.cc_id >= 0:
            config['tokenizer']['cc_id'] = tokenizer.cc_id
        if tokenizer.mask_id >= 0:
            config['tokenizer']['mask_id'] = tokenizer.mask_id
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        print(f"  <cc> token ID: {tokenizer.cc_id}")
        print(f"  <mask> token ID: {tokenizer.mask_id}")
    else:
        print("Warning: BPE tokenizer not found, using fallback tokenization")
    
    # Load word alignments from MFA if available
    word_alignments = None
    alignment_config = config.get('alignment', {})
    if alignment_config.get('use_alignment', False):
        alignment_path = alignment_config.get('word_alignments_path')
        word_alignments = load_word_alignments(alignment_path)
    
    if args.use_mix:
        # Use on-the-fly mixed training data
        # Default to truly on-the-fly mode (random mixing each epoch)
        truly_on_the_fly = data_config.get('truly_on_the_fly', True)  # Default to True for paper reproduction
        
        # Use train_manifest.json for on-the-fly mixing (it contains base samples)
        manifest_path = data_config.get('train_manifest', data_config.get('mix_train_manifest'))
        
        print("Loading mix training dataset...")
        if truly_on_the_fly:
            print("  Mode: Truly on-the-fly (random mixing each epoch, recommended for paper reproduction)")
        else:
            print("  Mode: Pre-generated mix configurations (for reproducibility)")
        
        # Get cc_id from tokenizer if available, otherwise from config
        cc_id = tokenizer.cc_id if tokenizer else config['tokenizer']['cc_id']
        
        train_dataset = LibriSpeechMixDataset(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            sample_rate=data_config['sample_rate'],
            n_mels=data_config['n_mels'],
            hop_length=data_config['hop_length'],
            win_length=data_config['win_length'],
            n_fft=data_config['n_fft'],
            min_duration=data_config['min_duration'],
            max_duration=data_config['max_duration'],
            overlap_ratio=data_config['overlap_ratio'],
            cc_id=cc_id,
            max_speakers=data_config.get('max_speakers', 2),
            truly_on_the_fly=truly_on_the_fly
        )
    else:
        # Use single speaker training data
        print("Loading single speaker training dataset...")
        train_dataset = LibriSpeechDataset(
            manifest_path=data_config['train_manifest'],
            tokenizer=tokenizer,
            word_alignments=word_alignments,
            sample_rate=data_config['sample_rate'],
            n_mels=data_config['n_mels'],
            hop_length=data_config['hop_length'],
            win_length=data_config['win_length'],
            n_fft=data_config['n_fft'],
            min_duration=data_config['min_duration'],
            max_duration=data_config['max_duration']
        )
    
    # Setup sampler for distributed training if needed
    if use_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = SASOTModel(
        vocab_size=config['model']['vocab_size'],
        num_speakers=config['model']['num_speakers'],
        asr_dim=config['model']['asr_dim'],
        spk_dim=config['model']['spk_dim'],
        cif_beta=config['model']['cif_beta']
    )
    model = model.to(device)
    
    # Verify speaker-aware attention is enabled (before DDP wrapping)
    if not use_distributed or local_rank == 0:
        from model.decoder.asr_decoder import TransformerDecoderLayer
        # Check if first layer uses speaker-aware attention
        first_layer = model.asr_decoder.layers[0]
        if hasattr(first_layer, 'use_speaker_aware') and first_layer.use_speaker_aware:
            print("✓ Speaker-aware attention is ENABLED in decoder")
        else:
            print("⚠️  WARNING: Speaker-aware attention is NOT enabled!")
        print()  # Empty line for readability
    
    # Wrap model for distributed training if needed
    if use_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not use_distributed or local_rank == 0:
        print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        if not use_distributed or local_rank == 0:
            print(f"\n📂 Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            # Handle DDP wrapping
            if use_distributed:
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get starting epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            if not use_distributed or local_rank == 0:
                print(f"  Resuming from epoch {start_epoch}")
        
        if not use_distributed or local_rank == 0:
            print("  ✓ Checkpoint loaded successfully\n")
    elif args.resume:
        if not use_distributed or local_rank == 0:
            print(f"⚠️  Warning: Resume checkpoint not found: {args.resume}")
            print("  Starting training from scratch...\n")
    
    # Speaker loss function
    speaker_loss_fn = SpeakerAMSoftmax(
        embed_dim=config['model']['spk_dim'],
        num_speakers=config['model']['num_speakers']
    )
    speaker_loss_fn = speaker_loss_fn.to(device)
    
    # Save initial checkpoint (epoch 0) before training starts
    if not use_distributed or local_rank == 0:
        print("\nSaving initial checkpoint (epoch 0)...")
        checkpoint_path = os.path.join(save_dir, 'checkpoint_epoch_0.pt')
        # Get model state dict (unwrap DDP if needed)
        model_state_dict = model.module.state_dict() if use_distributed else model.state_dict()
        torch.save({
            'epoch': 0,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.0,
            'config': config,
        }, checkpoint_path)
        print(f"  Initial checkpoint saved: {checkpoint_path}\n")
    
    # Training loop
    if not use_distributed or local_rank == 0:
        print("Starting training...\n")
        if start_epoch > 1:
            print(f"Continuing from epoch {start_epoch} to {config['training']['num_epochs']}\n")
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        if use_distributed:
            train_sampler.set_epoch(epoch)
        
        if not use_distributed or local_rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print("-" * 50)
        
        avg_loss, avg_asr_loss, avg_spk_loss = train_epoch(
            model, train_loader, optimizer, device,
            config['model']['vocab_size'], config['model']['num_speakers'],
            speaker_loss_fn, config
        )
        
        if not use_distributed or local_rank == 0:
            # Get loss weights for display
            asr_weight = config['training']['asr_loss_weight']
            spk_weight = config['training']['speaker_loss_weight']
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Total Loss (weighted): {avg_loss:.4f} = {asr_weight:.1f}×{avg_asr_loss:.4f} + {spk_weight:.1f}×{avg_spk_loss:.4f}")
            print(f"  ASR Loss: {avg_asr_loss:.4f} (weighted: {asr_weight * avg_asr_loss:.4f})")
            print(f"  Speaker Loss: {avg_spk_loss:.4f} (weighted: {spk_weight * avg_spk_loss:.4f})")
            print(f"  Note: Speaker loss may decrease slowly as it's an auxiliary task (weight={spk_weight})")
        
        # Save checkpoint every epoch (only on rank 0 for distributed training)
        if not use_distributed or local_rank == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            # Get model state dict (unwrap DDP if needed)
            model_state_dict = model.module.state_dict() if use_distributed else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model (only on rank 0 for distributed training)
    if not use_distributed or local_rank == 0:
        final_model_path = os.path.join(save_dir, 'final_model.pt')
        # Get model state dict (unwrap DDP if needed)
        model_state_dict = model.module.state_dict() if use_distributed else model.state_dict()
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, final_model_path)
        print(f"\nFinal model saved: {final_model_path}")
        print("\nTraining completed!")
    
    # Clean up distributed training
    if use_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
