# Snaption Training

This module contains training utilities for the Snaption image captioning model.

## Quick Start

### 1. Prepare Your Data

Download the Flickr8k dataset and organize it like this:
```
data/flickr8k/
├── captions.txt
└── Images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. Install Dependencies

```bash
pip install -e ..            # Install snaption package 
pip install tqdm matplotlib  # Additional training dependencies
```

### 3. Start Training

Basic + customized training:
```bash
# Inside the script, you may modify any settings you like.
python train.py
```

### 4. Resume Training

You may also resume training from a given checkpoint. You simply have to modify the `checkpoint_path` variable inside the `train.py` script.

## Training Configuration

### Model Parameters
- `context_length`: Maximum caption length (default: 20)
- `num_blocks`: Number of transformer layers (default: 6)
- `model_dim`: Model dimension (default: 512)
- `num_heads`: Number of attention heads (default: 16)
- `dropout_prob`: Dropout probability (default: 0.3)
- `encoder_model`: CNN encoder from timm (default: efficientnet_b0)

### Training Parameters
- `epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 2e-4)
- `weight_decay`: Weight decay for regularization (default: 5e-5)
- `label_smoothing`: Label smoothing factor (default: 0.1)
- `freeze_encoder`: Freeze CNN encoder weights (default: True)

### Data Augmentation

The training dataset applies several augmentations:
- Horizontal flipping (50% chance)
- Random resized crop (50% chance)
- Rotation, scaling, translation (50% chance)
- Color jittering (50% chance)  
- Brightness/contrast adjustment (30% chance)
- Gaussian blur (20% chance)

## Output Files

After training, you'll find these files in your save directory:

### Model Files
- `best_model.pt` - Best model based on validation loss
- `final_model.pt` - Final model after training
- `checkpoint_epoch_X.pt` - Periodic checkpoints

### Vocabulary Files
- `vocab_mapper.pkl` - Vocabulary mapper for inference
- `tokenizer.pkl` - Word-level text tokenizer

### Training Info
- `config.json` - Complete training configuration
- `training_history.json` - Loss curves and metrics
- `training_curves.png` - Training visualization

## Using Trained Models

```python
import pickle
import snaption

# Load vocabulary:
with open('checkpoints/vocab_mapper.pkl', 'rb') as f:
    vocab_mapper = pickle.load(f)

# Create model instance:
model = snaption.SnaptionModel(
    model_path = 'checkpoints/final_model.pt',
    vocab_mapper = vocab_mapper
)

# Generate caption:
caption = model.caption('path/to/image.jpg')
print(caption)
```

## Tips for Better Training

1. **Start Small**: Try training for 10 epochs first to ensure everything works
2. **Adjust Learning Rate**: If loss doesn't decrease, try a lower LR (1e-4)
3. **Use Validation**: It's encouraged to split your data to monitor overfitting
4. **Save Often**: Use the `save_every` variable in `train.py` for more frequent checkpoints

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 16, 8, or 1
- Use a smaller model: `model_dim = 256, num_heads = 8`

### Training Too Slow
- Increase `batch_size` if you have GPU memory
- Consider using a smaller Timm encoder: `encoder_model = efficientnet_b0`

### Poor Caption Quality
- Train for more epochs (300-500)
- Unfreeze encoder: `freeze_encoder = False`
- Increase model size: `model_dim = 768, num_heads = 12`
- Reduce `label_smoothing` to 0.05

## Advanced Usage

### Custom Dataset Format

If you have a different dataset format, modify `dataset.py`:

```python
# Add the following method to 'dataset.py'.
def load_custom_data(data_path):
    # Your custom data loading logic would go here.
    return df, all_captions
```

### Custom Model Architecture

Modify the model creation in `train.py` or extend the `ImageCaptioner` class.

