import sentencepiece as spm
import os


class BPETokenizer:
    """BPE Tokenizer using SentencePiece"""
    
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Get special token IDs
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        
        # Get <cc> and <mask> token IDs
        self.cc_id = self.sp.piece_to_id('<cc>')
        self.mask_id = self.sp.piece_to_id('<mask>')
        
        # If special tokens not found, use default IDs
        if self.cc_id == self.unk_id:
            self.cc_id = 3  # Default <cc> ID
        if self.mask_id == self.unk_id:
            self.mask_id = 0  # Default <mask> ID
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token IDs"""
        return self.sp.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos)
    
    def decode(self, ids):
        """Decode token IDs to text"""
        return self.sp.decode(ids)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.sp.get_piece_size()
    
    def id_to_piece(self, id):
        """Convert token ID to piece string"""
        return self.sp.id_to_piece(id)
    
    def piece_to_id(self, piece):
        """Convert piece string to token ID"""
        return self.sp.piece_to_id(piece)

