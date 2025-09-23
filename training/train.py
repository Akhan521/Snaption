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

    # Create train/validation split:
    print("\nCreating train/validation split...")
    val_split = 0.0 # We'll use the entire dataset for training.
    if val_split > 0:
        train_df, val_df = create_train_val_split(df, val_split = val_split)
        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    else:
        train_df, val_df = df, None
        print(f"Training samples: {len(train_df)}, No validation set provided.")

    # Create datasets:
    print("\nCreating datasets...")
    train_dataset = ImageCaptioningDataset(
        train_df, images_dir, split = 'train',
        tokenizer = tokenizer, vocab_mapper = vocab_mapper
    )

    if val_df is not None:
        val_dataset = ImageCaptioningDataset(
            val_df, images_dir, split = 'val',
            tokenizer = tokenizer, vocab_mapper = vocab_mapper
        )
    else:
        val_dataset = None

    # Create data loaders:
    print("\nCreating data loaders...")
    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
    )

    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = False,
        )
    else:
        val_dataloader = None

    print(f"Train batches: {len(train_dataloader)}")
    if val_dataloader is not None:
        print(f"Validation batches: {len(val_dataloader)}")
    else:
        print("No validation dataloader available.")

    # Our model configuration:
    context_length = 20 # Max caption length (including special tokens).
    num_blocks = 6      # Number of transformer blocks.
    model_dim = 512     # Model dimensionality.
    num_heads = 16      # Number of attention heads.
    dropout_prob = 0.5  # Dropout probability.
    encoder_model = 'efficientnet_b0' # Timm model name for encoder.

    # Create model:
    print("\nCreating model...")
    model = snaption.ImageCaptioner(
        context_length = context_length,
        vocab_size = len(vocab_mapper),
        num_blocks = num_blocks,
        model_dim = model_dim,
        num_heads = num_heads,
        dropout_prob = dropout_prob,
        encoder_model = encoder_model # You can change the encoder model here.
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create trainer:
    print("\nCreating and setting up trainer...")
    trainer = SnaptionTrainer(
        model = model,
        train_loader = train_dataloader,
        val_loader = val_dataloader,
        vocab_mapper = vocab_mapper,
        device = device,
    )

    # Our training configuration:
    learning_rate = 2e-4
    weight_decay = 1e-4
    label_smoothing = 0.1 # This helps regularize the model.
    max_epochs = 10       # Set to a small number for testing; increase as needed.
    freeze_encoder = True # Whether to freeze the CNN encoder during training.
    use_scheduler = True  # Whether to use a learning rate scheduler.

    # Set up trainer:
    trainer.setup_training(
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        label_smoothing = label_smoothing,
        max_epochs = max_epochs, 
        freeze_encoder = freeze_encoder,
        use_scheduler = use_scheduler,
    )