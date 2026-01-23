"""Load word alignments from MFA output"""
import json
import os


def load_word_alignments(alignment_path):
    """
    Load word emission times from MFA alignment JSON file
    
    Args:
        alignment_path: Path to word_alignments.json file
        
    Returns:
        dict: Mapping from utterance_id to list of word alignments
              Each alignment: {'word': str, 'start': float, 'end': float, 'duration': float}
    """
    if alignment_path is None or not os.path.exists(alignment_path):
        print(f"Warning: Alignment file not found: {alignment_path}")
        return None
    
    with open(alignment_path, 'r', encoding='utf-8') as f:
        alignments = json.load(f)
    
    print(f"Loaded word alignments for {len(alignments)} utterances")
    return alignments

