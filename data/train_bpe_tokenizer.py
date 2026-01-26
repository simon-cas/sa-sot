#!/usr/bin/env python3
"""
Train BPE tokenizer using SentencePiece for SA-SOT model
"""

import os
import argparse
import sentencepiece as spm
from pathlib import Path


def collect_texts_from_manifest(manifest_path):
    """Collect all transcriptions from manifest file"""
    import json
    
    texts = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    samples = manifest.get('samples', [])
    for sample in samples:
        transcription = sample.get('transcription', '').strip()
        if transcription:
            texts.append(transcription)
    
    return texts


def train_bpe_tokenizer(text_file, output_model_prefix, vocab_size=3730, 
                        character_coverage=0.9995, model_type='bpe'):
    """
    Train SentencePiece BPE tokenizer
    
    Args:
        text_file: Input text file (one sentence per line)
        output_model_prefix: Output model prefix (will generate .model and .vocab)
        vocab_size: Vocabulary size
        character_coverage: Character coverage (0.0-1.0)
        model_type: 'bpe' or 'unigram'
    """
    print(f"Training {model_type.upper()} tokenizer...")
    print(f"  Input: {text_file}")
    print(f"  Output prefix: {output_model_prefix}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Character coverage: {character_coverage}")
    
    # SentencePiece training parameters
    # Note: byte_fallback=True creates byte-level tokens (<0x00> to <0xFF>) for unknown characters
    # These tokens have score=0 because they're rarely used, but they're necessary for robustness
    # For English text, we can set byte_fallback=False to avoid these tokens and use more vocab for words
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=output_model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,  # 'bpe' or 'unigram'
        input_sentence_size=1000000,  # Limit input sentences
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1000000,
        shrinking_factor=0.75,
        num_threads=os.cpu_count(),
        num_sub_iterations=2,
        max_sentence_length=4192,
        split_by_unicode_script=True,
        split_by_whitespace=True,  # Split by whitespace (spaces become ▁ prefix)
        split_by_number=True,
        treat_whitespace_as_suffix=False,  # Whitespace becomes prefix (▁) not suffix
        # For English text, byte_fallback can be False to save vocab space
        # Set to True only if you need to handle rare/unknown characters
        byte_fallback=False,  # Disable byte fallback for English (saves ~256 vocab slots)
        vocabulary_output_piece_score=True,
        hard_vocab_limit=True,
        use_all_vocab=False,
        unk_id=0,  # <unk> token ID
        bos_id=1,  # <s> token ID
        eos_id=2,  # </s> token ID
        pad_id=-1,  # <pad> token ID (not used in SentencePiece)
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=['<cc>', '<mask>', '<pad>'],  # Special symbols for SA-SOT
    )
    
    print(f"\n✓ Tokenizer training completed!")
    print(f"  Model: {output_model_prefix}.model")
    print(f"  Vocab: {output_model_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer using SentencePiece')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to training manifest JSON file')
    parser.add_argument('--output_dir', type=str, default='./data/tokenizer',
                       help='Output directory for tokenizer model')
    parser.add_argument('--vocab_size', type=int, default=3730,
                       help='Vocabulary size')
    parser.add_argument('--character_coverage', type=float, default=0.9995,
                       help='Character coverage (0.0-1.0)')
    parser.add_argument('--model_type', type=str, default='bpe',
                       choices=['bpe', 'unigram'],
                       help='SentencePiece model type')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect texts from manifest
    print("Collecting transcriptions from manifest...")
    texts = collect_texts_from_manifest(args.manifest)
    print(f"Found {len(texts)} transcriptions")
    
    if len(texts) == 0:
        print("Error: No transcriptions found in manifest!")
        return
    
    # Write texts to temporary file
    temp_text_file = os.path.join(args.output_dir, 'temp_texts.txt')
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    print(f"Wrote {len(texts)} sentences to {temp_text_file}")
    
    # Train tokenizer
    output_model_prefix = os.path.join(args.output_dir, 'bpe_tokenizer')
    train_bpe_tokenizer(
        temp_text_file,
        output_model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type
    )
    
    # Clean up temporary file
    os.remove(temp_text_file)
    print(f"\nCleaned up temporary file: {temp_text_file}")
    
    print(f"\n✓ BPE tokenizer training completed!")
    print(f"  Model: {output_model_prefix}.model")
    print(f"  Vocab: {output_model_prefix}.vocab")
    print(f"\nUse this model path in config.yaml: {output_model_prefix}.model")


if __name__ == '__main__':
    main()

