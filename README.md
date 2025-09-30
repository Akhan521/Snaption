# Snaption 📸
### AI-Powered Image Captioning with PyTorch, Transformers, and Multi-Modal Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Transform your images into natural language descriptions using deep learning**

[🚀 Quick Start](#-quick-start) • [📦 Releases](https://github.com/Akhan521/Snaption/releases/tag/v0.1.0) • [📖 Docs](#-documentation) • [🎯 Demo](#-demo)

---

## 🎬 What is Snaption?

Snaption is an end-to-end **image captioning system** that automatically generates natural language descriptions for images. Built entirely from scratch using PyTorch, it represents my journey into multi-modal deep learning, combining computer vision and natural language processing in a single, cohesive system.

**Key Highlights:**
- **Transformer-based architecture** with EfficientNet encoder and custom decoder
- **Production-ready Python package** with clean API and comprehensive documentation  
- **Pre-trained models** available for immediate use
- **Extensible training pipeline** with advanced data augmentation
- **Professional MLOps practices** including checkpointing, monitoring, and reproducibility

**Potential Real-World Applications:**
- Accessibility tools for visually impaired users
- Automatic image tagging for content management
- E-commerce product descriptions

---

## ✨ Demo

**Input Images → Generated Captions**

```python
import snaption
import pickle

# Load vocabulary:
with open('vocab_mapper.pkl', 'rb') as f:
    vocab_mapper = pickle.load(f)

# Create model instance with pre-trained weights:
model = snaption.SnaptionModel('path/to/pretrained_weights.pt', vocab_mapper)

# Caption any image:
caption = model.caption('path/to/your/image.jpg')

# Example outputs from the model:
```

| Image | Generated Caption |
|-------|-------------------|
| 🐕 Dog in park | "a dog is running through the grass" |
| 🏖️ Beach scene | "a man is standing on the beach near the ocean" |
| 👶 Child playing | "a young child is playing with a toy" |

> **Note**: Captions are generated in real-time from the model's learned understanding of images and language!

---

## Quick Start

### Option A: Use Pre-Trained Models (Recommended)

**Get up and running in 5 minutes:**

```bash
# 1. Clone and install:
git clone https://github.com/Akhan521/Snaption.git
cd snaption
pip install -e .

# 2. Download pre-trained models from Releases:
# https://github.com/Akhan521/Snaption/releases/tag/v0.1.0
# Download all provided files (model weights, vocab mapper).
```

```python
# 3. Start captioning!
import snaption
import pickle

# Load vocabulary:
with open('vocab_mapper.pkl', 'rb') as f:
    vocab_mapper = pickle.load(f)

# Create model:
model = snaption.SnaptionModel(
    model_path='path/to/pretrained_weights.pt',
    vocab_mapper=vocab_mapper
)

# Caption any image:
caption = model.caption('path/to/your/image.jpg')
print(f"Caption: {caption}")
```

### Option B: Train Your Own Model

```bash
# 1. Setup dataset (see DATASET_SETUP.md)

# 2. Start training:
cd training/
# Feel free to adjust hyperparameters in train.py if desired or use defaults.
python train.py 
```

**📖 Detailed guides**: [Installation](#installation) • [Training Guide](training/README.md) • [Dataset Setup](DATASET_SETUP.md)

---

## 🏗️ Architecture & Design

Snaption uses an **encoder-decoder architecture** specifically designed for image captioning:

### Model Architecture

```
Image (224×224×3)
    ↓
┌─────────────────────┐
│  EfficientNet-B0    │  ← Pre-trained CNN Encoder
│  (Frozen)           │    Extracts visual features
└─────────────────────┘
    ↓ (1280-dim features)
┌─────────────────────┐
│  Linear Projection  │  ← Project to model dimension
└─────────────────────┘
    ↓ (512-dim)
┌─────────────────────┐
│  Transformer        │  ← 6-layer decoder with
│  Decoder            │    self-attention + cross-attention
│  (6 layers)         │    Generates caption tokens
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Vocabulary         │  ← Final projection to vocab
│  Projection         │    (~8000 tokens)
└─────────────────────┘
    ↓
Generated Caption: "a dog is running through the grass"
```

### Design Decisions

**Why EfficientNet-B0?**
- Best accuracy/efficiency trade-off for edge deployment
- Pre-trained on ImageNet for robust feature extraction
- Only 5.3M parameters (vs 25M+ for ResNet-50)
- Frozen during training to prevent overfitting on small dataset

**Why 6-layer Transformer?**
- Balances model capacity with training time
- Sufficient for learning caption patterns in Flickr8k
- Self-attention captures word dependencies
- Cross-attention grounds language in visual features

**Key Hyperparameters:**
- Model dimension: 512 (split across 16 attention heads)
- Context length: 20 tokens (max caption length)
- Vocabulary size: ~8,000 tokens
- Dropout: 0.5 (for regularization)

---

## 📊 Results & Performance

### Training Progress

The model was trained on **Flickr8k** (8,091 images, 40,455 captions):

- **Training epochs**: 500
- **Final training loss**: ~2.15
- **Training time**: ~6-8 hours on a high-ram T4 GPU

### Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Model Size** | 87 MB | Compressed state dict |
| **Inference Speed** | ~1s/image | On CPU (Intel i7-1365U) |
| **Vocabulary** | ~8,000 tokens | Covers 99% of the dataset |

### Sample Outputs

**Strong Examples:**
- Input: Beach scene → Output: "a man is standing on the beach"
- Input: Dog playing → Output: "a dog is running through the grass"
- Input: Street scene → Output: "a man is riding a bike down the street"

**Limitations:**
- Struggles with complex scenes (multiple objects)
- Limited vocabulary (e.g., rare words may be missed)
- Sometimes generic descriptions (e.g., "a dog is on the grass")
- Trained only on Flickr8k → may be biased towards common scenes

**Potential future improvements**: Fine-tune on larger datasets (MS COCO), implement beam search, add attention visualization

---

## 💡 My Learning Journey

### Motivation

I wanted to explore multi-modal ML for the first time, and I decided that diving into the intersection of computer vision and natural language processing would be the perfect challenge, as these are both fields I’m passionate about. This led me to image captioning. Image captioning is technically complex, has real-world applications, and forces you to understand both modalities deeply.

### What I Learned

**Before this project:**
- Basic multi-modal ML knowledge from personal study
- Some PyTorch experience with simple CNNs
- Simpler understanding of Transformers

**After this project:**
- **Deep Learning**: Hands-on experience with the Transformer architecture, attention mechanisms, and multi-modal learning
- **Computer Vision**: CNN feature extraction, transfer learning, image preprocessing, data augmentation strategies
- **NLP**: Tokenization, vocabulary building, sequence generation, autoregressive decoding
- **MLOps**: Training pipelines, hyperparameter tuning, checkpointing, loss monitoring, reproducibility
- **Software Engineering**: Building installable packages, API design, documentation, version control
- **Debugging**: Identifying overfitting, diagnosing training instabilities, validating data pipelines

### Key "Aha!" Moments

**1. Understanding Cross-Attention**
> Initially, I struggled to understand how the model "sees" the image while generating text. The breakthrough came when I realized cross-attention creates a bridge: the decoder's query vectors ask "what visual features are relevant for generating this word?", and the image features respond. It's like the model is looking back at the image for each word it writes.

**2. Why Freezing the Encoder Matters**
> Freezing EfficientNet seemed counter-intuitive since I thought, "Aren't more trainable parameters better?" But with only 8K images, the pre-trained features were already excellent, and fine-tuning just caused memorization. This taught me: **more parameters ≠ better performance** on small datasets.

**3. The Power of Data Augmentation**
> Adding aggressive augmentations (random crops, color jitter, rotations) improved feature generalization. The model was no longer seeing the exact same image twice; it learned to focus on semantic content rather than pixel-level details. This was my first real experience with regularization that actually works in practice.

### Resources That Helped

- **"Attention is All You Need"** paper for Transformer fundamentals
- **PyTorch tutorials** for encoder-decoder implementations
- **Andrej Karpathy's blog** on RNNs and language modeling

---

