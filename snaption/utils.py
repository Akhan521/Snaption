import re
import torch
from collections import Counter
from typing import List, Dict, Tuple

class WordTokenizer:
    '''Simple word tokenizer that extracts words from text.'''

    def __init__(self):
        self.pattern = re.compile(r'\b\w+\b')

    def __call__(self, text: str) -> List[str]:
        '''Tokenize text into lowercase words.'''
        return self.pattern.findall(text.lower())
    
    def tokenize(self, text: str) -> List[str]:
        '''Alias for __call__ method.'''
        return self(text)
    
class VocabMapper:
    '''
    Vocabulary mapper that converts tokens to indices and vice versa.

    Special tokens:
    - <UNKNOWN>: 0 (for out-of-vocabulary words)
    - <PAD>:     1 (for padding sequences)
    - <START>:   2 (beginning of sequence)
    - <END>:     3 (end of sequence)
    '''

    def __init__(self, vocab_freqs: Dict[str, int], min_freq: int = 1):
        '''
        Initialize our vocabulary mapper.

        Args:
            vocab_freqs (Dict[str, int]): Dictionary of token frequencies
            min_freq (int): Minimum frequency threshold for including tokens
        '''
        # Initialize token-to-index mapping.
        self.token_to_idx = {
            '<UNKNOWN>': 0,
            '<PAD>': 1,
            '<START>': 2,
            '<END>': 3
        }

        # For tokens that meet the min_freq threshold, add them to the mapping.
        idx = 4
        for token, freq in vocab_freqs.items():
            if freq >= min_freq:
                self.token_to_idx[token] = idx
                idx += 1

        # Create reverse mapping.
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.default_token = '<UNKNOWN>'
        self.default_idx = self.token_to_idx[self.default_token]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_idx)
    
    def __getitem__(self, token: str) -> int:
        """Get index for a single token."""
        return self.token_to_idx.get(token, self.default_idx)
    
    def __call__(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to list of indices."""
        return [self[token] for token in tokens]
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Alias for __call__ method."""
        return self(tokens)

    def decode(self, indices: List[int] | torch.Tensor) -> str:
        '''
        Convert a list of indices back to text.

        Args:
            indices (List[int] | torch.Tensor): List of indices to decode.

        Returns:
            str: Decoded text.
        '''
        tokens = []

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        for idx in indices:
            token = self.idx_to_token.get(idx, self.default_token)
            tokens.append(token)

            if token == '<END>':
                break

        return ' '.join(tokens)
    
    def decode_batch(self, batch_indices: torch.Tensor) -> List[str]:
        '''
        Decode a batch of sequences.

        Args:
            batch_indices (torch.Tensor): Batch of indices to decode of shape (batch_size, seq_length).

        Returns:
            List[str]: List of decoded strings for each sequence in the batch.
        '''
        return [self.decode(seq) for seq in batch_indices]
    
    def get_special_tokens(self) -> Dict[str, int]:
        '''Return dictionary of special tokens and their indices'''
        return {
            'unknown': self.token_to_idx['<UNKNOWN>'],
            'pad': self.token_to_idx['<PAD>'], 
            'start': self.token_to_idx['<START>'],
            'end': self.token_to_idx['<END>']
        }

def build_vocab(captions: List[str], min_freq: int = 1) -> Tuple[WordTokenizer, VocabMapper]:
    '''
    Build tokenizer and vocab mapper from a list of captions.

    Args:
        captions (List[str]): List of caption strings
        min_freq (int): Minimum frequency threshold for including tokens

    Returns:
        Tuple[WordTokenizer, VocabMapper]: The tokenizer and vocabulary mapper
    '''
    tokenizer = WordTokenizer()
    vocab_freqs = Counter()

    # Tokenize each caption and update vocabulary frequencies.
    for caption in captions:
        tokens = tokenizer(caption)
        vocab_freqs.update(tokens)

    # Create vocabulary mapper.
    vocab_mapper = VocabMapper(vocab_freqs, min_freq = min_freq)

    return tokenizer, vocab_mapper

def prepare_caption(
    caption: str,
    tokenizer: WordTokenizer,
    vocab_mapper: VocabMapper,
    max_length: int
) -> torch.Tensor:
    '''
    Prepare a single caption for training or inference.

    Args:
        caption (str): Raw caption string
        tokenizer (WordTokenizer): Word-level tokenizer
        vocab_mapper (VocabMapper): Vocabulary mapper
        max_length (int): Maximum sequence length

    Returns:
        torch.Tensor: Tensor of token indices with padding/truncation.
    '''
    # Tokenize caption and convert to indices.
    tokens = tokenizer(caption)
    encoded = vocab_mapper(tokens)

    # Add start and end tokens.
    encoded = [vocab_mapper['<START>']] + encoded + [vocab_mapper['<END>']]

    # If we need to pad our sequence, we'll do so.
    if len(encoded) < max_length:
        pad_len = max_length - len(encoded)
        encoded += [vocab_mapper['<PAD>']] * pad_len
    # Otherwise, we'll truncate our sequence and add an end token.
    else:
        encoded = encoded[:max_length - 1] + [vocab_mapper['<END>']]

    return torch.tensor(encoded, dtype=torch.long)

def cleanup_caption(caption: str) -> str:
    '''
    Clean-up a generated caption by removing special tokens and extra whitespace.

    Args:
        caption (str): Generated caption string.

    Returns:
        str: Cleaned-up caption.
    '''
    special_tokens = ['<UNKNOWN>', '<PAD>', '<START>', '<END>']
    for token in special_tokens:
        caption = caption.replace(token, '')

    # Remove extra whitespace.
    caption = ' '.join(caption.split())
    
    # Capitalize the first letter.
    if caption:
        caption = caption[0].upper() + caption[1:]

    return caption.strip()
