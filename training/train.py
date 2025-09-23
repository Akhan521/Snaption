'''
Main training script for Snaption image-captioning model.

Usage:
    python training/train.py

Note:
- Ensure you have the required dataset (e.g., Flickr8k) downloaded and placed in the correct directory (see paths below).
- Ensure that all provided paths are correct before running the script.
- Adjust model and training parameters as needed in the script.
- The script saves model checkpoints and the training configuration to the specified save directory.
- You can interrupt training with Ctrl + C; the current model state will be saved.
- You can resume training from a saved checkpoint by specifying the checkpoint path in the script.

'''

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import pickle

import snaption
from dataset import load_flickr8k_dataset, create_train_val_split, ImageCaptioningDataset
from trainer import SnaptionTrainer

def setup_device(device: str = 'auto') -> str:
    '''Set up the device for training.'''
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu', 'cuda', or 'auto'")
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine")
        
    print(f"Using device: {device}")
    return device

def main():
    '''Main function set up training.'''
    # Set up device:
    device = setup_device('auto')

    # Set up paths (ensure these paths are correct):
    data_dir = Path('data/Flickr8k')
    captions_file = data_dir / 'captions.txt'
    images_dir = data_dir / 'images'
    save_dir = Path('./checkpoints')
    # Create save directory if it doesn't exist:
    save_dir.mkdir(parents = True, exist_ok = True)

    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")

    # Load dataset:
    print("\nLoading dataset...")
    df, all_captions = load_flickr8k_dataset(captions_file, images_dir)

    # Build vocabulary:
    print("\nBuilding vocabulary...")
    tokenizer, vocab_mapper = snaption.utils.build_vocab(all_captions, min_freq = 2)
    print(f"Vocabulary size: {len(vocab_mapper)}")

    # Save tokenizer and vocab mapper for future use:
    vocab_path = save_dir / 'vocab_mapper.pkl'
    tokenizer_path = save_dir / 'tokenizer.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_mapper, f)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"Vocab mapper saved to: {vocab_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")

    