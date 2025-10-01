# Snaption 📸: AI-Powered Image Captioning 

<div align="center">

### AI-Powered Image Captioning with PyTorch, Deep Learning, & Multi-Modal ML
**Transform your images into natural language descriptions using deep learning!**

[![Quick Start](https://img.shields.io/badge/🚀_Quick_Start-4facfe?style=for-the-badge)](#-quick-start)
[![Models](https://img.shields.io/badge/📦_Models-blue?style=for-the-badge)](https://github.com/Akhan521/Snaption/releases/tag/v0.1.0)
[![Demo](https://img.shields.io/badge/🎯_Demo-667eea?style=for-the-badge)](#-demo)
[![Docs](https://img.shields.io/badge/📖_Docs-9B59B6?style=for-the-badge)](#-documentation)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Install](https://img.shields.io/badge/Install-Source_Only-orange.svg)](https://github.com/Akhan521/Snaption#-quick-start)

</div>


---

## 🎬 What is Snaption?

Snaption is an end-to-end **image captioning system** that automatically generates natural language descriptions for images. Built entirely from scratch using PyTorch, it represents my journey into multi-modal deep learning, combining computer vision and natural language processing in a single, cohesive system.

**Key Highlights:**
- **Transformer-based architecture** with EfficientNet encoder and custom decoder
- **Production-ready Python package** with clean API and comprehensive documentation  
- **Pre-trained models** available for immediate use
- **Extensible training pipeline** with advanced data augmentation
- **Professional MLOps practices** including checkpointing, monitoring, and reproducibility
> **📦 Installation Note**: Snaption is intentionally distributed as a source installation project rather than a PyPI package. This design choice prioritizes code transparency, easy modification, and focuses on demonstrating ML engineering skills over package distribution. My project structure follows production-ready packaging standards and could be published to PyPI, if extended for broader use.

**Potential Real-World Applications:**
- Accessibility tools for visually impaired users
- Automatic image tagging for content management
- E-commerce product descriptions

---

## ✨ Demo

<div align="center">


https://github.com/user-attachments/assets/19dad98f-fbfb-474b-b75a-38446732a9ae


</div>

### Example outputs from the model:

| Image | Generated Caption |
|-------|-------------------|
| 🐕 Dog in park | "a dog is running through the grass" |
| 🏖️ Beach scene | "a man is standing on the beach near the ocean" |
| 👶 Child playing | "a young child is playing with a toy" |

**Note**: Captions are generated in real-time from the model's learned understanding of images and language!

---

## 🚀 Quick Start

### Option A: Use Pre-Trained Models (Recommended)

**Get up and running in 5 minutes:**

```bash
# 1. Clone and install (source installation):
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
# 1. Clone and install (source installation):
git clone https://github.com/Akhan521/Snaption.git
cd snaption
pip install -e .

# 2. Set up dataset: (see DATASET_SETUP.md)

# 3. Start training: Feel free to adjust hyperparameters in train.py if desired or use defaults.
python -m training.train
```

**📖 Detailed guides**: [Training Guide](training/README.md) || [Dataset Setup](DATASET_SETUP.md)

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

### Design Decisions:

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
- Dropout: 0.3 (for regularization)

---

## 📊 Results & Performance

### Training Progress:

The model was trained on **Flickr8k** (8,091 images, 40,455 captions):

- **Training epochs**: 500
- **Final training loss**: ~2.15
- **Training time**: ~6-8 hours on a high-ram T4 GPU

### Model Performance:

| Metric | Value | Notes |
|--------|-------|-------|
| **Model Size** | 87 MB | Compressed state dict |
| **Inference Speed** | ~1s/image | On CPU (Intel i7-1365U) |
| **Vocabulary** | ~8,000 tokens | Covers 99% of the dataset |

### Sample Outputs:

**Strong Examples:**
- Input: Beach scene → Output: "a man is standing on the beach"
- Input: Dog playing → Output: "a dog is running through the grass"
- Input: Street scene → Output: "a man is riding a bike down the street"

**Limitations:**
- Struggles with complex scenes (multiple objects)
- Limited vocabulary (e.g., rare words may be missed)
- Sometimes generic descriptions (e.g., "a dog is on the grass")
- Trained only on Flickr8k → may be biased towards common scenes

**Potential improvements**: Fine-tune on larger datasets (MS COCO), add attention visualization, etc.

---

## 💡 My Learning Journey

### Motivation:

I wanted to explore multi-modal ML for the first time, and I decided that diving into the intersection of computer vision and natural language processing would be the perfect challenge, as these are both fields I’m passionate about. This led me to image captioning. Image captioning is technically complex, has real-world applications, and forces you to understand both modalities deeply.

### What I Learned:

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

### Key "Aha!" Moments:

**1. Understanding Cross-Attention**
> Initially, I struggled to understand how the model "sees" the image while generating text. The breakthrough came when I realized cross-attention creates a bridge: the decoder's query vectors ask "what visual features are relevant for generating this word?", and the image features respond. It's like the model is looking back at the image for each word it writes.

**2. Why Freezing the Encoder Matters**
> Freezing EfficientNet seemed counter-intuitive since I thought, "Aren't more trainable parameters better?" But with only 8K images, the pre-trained features were already excellent, and fine-tuning just caused memorization. This taught me: **more parameters ≠ better performance** on small datasets.

**3. The Power of Data Augmentation**
> Adding aggressive augmentations (random crops, color jitter, rotations) improved feature generalization. The model was no longer seeing the exact same image twice; it learned to focus on semantic content rather than pixel-level details. This was my first real experience with regularization that actually works in practice.

### Resources That Helped:

- **"Attention is All You Need"** paper for Transformer fundamentals
- **PyTorch tutorials** for encoder-decoder implementations
- **Andrej Karpathy's blog** on RNNs and language modeling

---

## 🚧 Challenges & Solutions

### Challenge 1: Finding the Right Hyperparameters

**The Problem:**  
My initial training runs showed unstable loss curves, sometimes diverging, other times barely moving. I tried dozens of configurations: different learning rates (1e-3 to 2e-5), batch sizes (8 to 32), model dimensions (256 to 512). My results weren't exactly improving as expected.

**My Approach:**
1. **Systematic debugging**: Started with a tiny model (2 layers, 128 dim) on 100 images to verify the pipeline worked
2. **Learning rate search**: Conducted a learning rate range test (1e-6 to 2e-4) to find a stable starting point
3. **Scheduler experimentation**: Tested constant LR, step decay, warm-up + cosine annealing, and OneCycleLR

**The Solution:**  
My OneCycleLR scheduler setup with warm-up was the game-changer. Starting with a small LR, warming up to 2e-4 over 10% of training, then cosine annealing to near-zero. This gave my model time to stabilize early on, then converge smoothly. Gradient clipping (max norm 2.0) prevented occasional spikes.

**Result**: Stable training with steady loss decrease from +10 → ~2.15 over 500 epochs.

**Key Learning**: Hyperparameter tuning isn't random guessing, it's a systematic debugging process. Start simple, isolate variables, and use diagnostic tools.

---

### Challenge 2: Overfitting on a Small Dataset

**The Problem:**  
Flickr8k has only 8,091 images, which is tiny by deep learning standards. Initially, my model was memorizing captions instead of learning to describe images. What I found was that generated captions were sometimes exact copies of training captions for different images!

**My Approach:**
1. **Research phase**: Read content on training with limited data (data augmentation, regularization techniques)
2. **Experimentation matrix**: Tested multiple strategies systematically
   - Dropout rates: 0.1, 0.3, 0.5
   - Data augmentations: basic (flip) → aggressive (crop, rotate, color jitter)
   - Label smoothing: 0, 0.05, 0.1, 0.2

**The Solution:**  
A combination approach worked best for me:
- **Froze the encoder**: Reduced trainable parameters for better generalization on a small dataset
- **Aggressive augmentations**: Random crops, color jitter, horizontal flips, rotations
- **Label smoothing (0.1)**: Prevented overconfidence in predictions
- **Dropout (0.3)**: Applied to all Transformer layers

**Result**: Generated captions became more generalizable and diverse.

**Key Learning**: With small datasets, **preventing overfitting is as important as model architecture**. Regularization isn't optional, I would argue that it's essential.

---

### Challenge 3: Data Preprocessing Pipeline

**The Problem:**  
How much augmentation is too much? I wanted robust training, but I didn't want to corrupt the images so much that captions became incorrect or unreliable (e.g., flipping a "left" to "right", or color jittering a "red car" into unrecognizable colors).

**My Approach:**
1. **Visual inspection**: Created a Colab notebook to preview augmented images alongside captions
2. **Probability tuning**: Started with 100% augmentation probability, gradually reduced until images looked reasonable
3. **Semantic consistency**: Avoided transformations that change meaning (e.g., no vertical flips for most scenes)

**The Solution:**  
My final augmentation pipeline with reasonable probabilities:
- Horizontal flip: 50% (people/animals look natural either way)
- Random resized crop: 50% (scale: 0.85-1.0, ratio: 0.9-1.1)
- Rotation: 50% (max ±15° to avoid disorientation)
- Color jitter: 50% (brightness/contrast/saturation: ±20%)
- Gaussian blur: 20% (mild, for robustness)

**Result**: My model learned to handle variations without learning incorrect associations.

**Key Learning**: Data augmentation requires **domain knowledge and visual intuition**. It's not just applying every transformation available. Less can be more if chosen thoughtfully.

---

### Challenge 4: Building a Production-Ready Package

**The Problem:**  
My initial code was a Colab notebook which was great for prototyping, but not reusable or shareable. I wanted to create something others could actually install and use, but I'd never built a Python package before.

**My Approach:**
1. **Research**: Looked to popular ML packages (Hugging Face Transformers, timm) for best practices and inspiration
2. **Refactoring**: Separated concerns (model definition, inference, training, data loading) into different modules
3. **API design**: Thought about user experience and considered what should be simple vs. customizable?
4. **Documentation**: Wrote docstrings, type hints, and README guides
5. **Testing**: Created a test script to ensure everything worked as expected

**The Solution:**  
Clean package structure with:
- `snaption/` core package (model, inference, utils)
- `training/` optional training utilities (not required for inference)
- `setup.py` for pip installation
- Comprehensive documentation (README, dataset guide, training guide)

**Result**: Users can `pip install snaption` and start captioning in ~5 lines of code. Training code is available but separate for advanced users.

**Key Learning**: **Good software engineering makes ML content accessible**. It's just as much about usability as it is about model performance.

---

## 🛠️ Technical Stack

**Core Framework:**
- **PyTorch 2.8+** - Deep learning framework
- **Python 3.12+** - Programming language

**Computer Vision:**
- **timm** - Pre-trained vision models (EfficientNet)
- **OpenCV** - Image processing
- **Albumentations** - Advanced data augmentation
- **Pillow** - Image I/O

**NLP & Data:**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Custom tokenizer** - Text preprocessing

**Training:**
- **AdamW** - Optimizer with weight decay
- **OneCycleLR** - Learning rate scheduler

**Development Tools:**
- **Git/GitHub** - Version control
- **Google Colab** - Prototyping and GPU access for training
- **tqdm** - Progress monitoring
- **Matplotlib** - Visualization

**Package & Distribution:**
- **setuptools** - Package building
- **pip** - Distribution
- **GitHub Releases** - Model hosting

---

## 📖 Documentation

- **[Quick Start Guide](#-quick-start)** - Get running in 5 minutes
- **[Dataset Setup](DATASET_SETUP.md)** - Download Flickr8k dataset
- **[Training Guide](training/README.md)** - Train your own models

---
## 💭 Installation & Distribution Philosophy

### Not on PyPI?

Snaption is structured as a **production-ready Python package** but intentionally maintained as a source installation project. This design choice reflects:

**Strategic Benefits:**
- **Code Transparency**: Users can easily explore and understand my full implementation
- **Modification Freedom**: Researchers and learners can adapt my code for their needs
- **Focus on Learning**: Emphasizes code quality and architecture over distribution metrics
- **No Maintenance Burden**: Allows me to focus on new projects without ongoing package support obligations

**Package Maturity:**
- Proper `setup.py` configuration following Python packaging standards
- Clean modular structure separating core package from training utilities
- Comprehensive documentation and type hints
- Ready for PyPI publication if extended for production use

**Installation remains simple** - it takes just 3 commands to get started. This demonstrates my packaging skills while keeping my project scope appropriate for a portfolio piece.

---

## 🤝🏼 Contributing

I welcome contributions! From fixing bugs, adding features, or improving documentation, here's how you can help:

### Ways to Contribute:

1. **Report bugs**: [Open an issue](https://github.com/Akhan521/Snaption/issues) with reproducible steps
2. **Suggest features**: Share ideas for improvements
3. **Improve docs**: Fix typos, add examples, clarify explanations
4. **Submit PRs**: Fork, create a branch, and submit a pull request

### Development Setup:

```bash
# Clone and install in development mode:
git clone https://github.com/Akhan521/Snaption.git
cd snaption
pip install -e .

# Run tests: 
python -m tests/snaption_tests.py
# (Add more tests as needed...)
```

### Contribution Guidelines:

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include type hints
- Write tests for new features
- Update documentation

---

## 📚 References & Acknowledgments

### Papers & Research:

- **Vaswani et al. (2017)**: ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) - Transformer architecture
- **Xu et al. (2015)**: ["Show, Attend and Tell"](https://arxiv.org/abs/1502.03044) - Attention for image captioning
- **Tan & Le (2019)**: ["EfficientNet"](https://arxiv.org/abs/1905.11946) - Efficient CNN architecture

### Datasets:

- **Flickr8k**: Hodosh et al. (2013) - ["Framing Image Description as a Ranking Task"](https://doi.org/10.1613/jair.3994)

### Tools & Libraries:

- **PyTorch Team** - For the excellent deep learning framework
- **Ross Wightman** - For the timm library (pre-trained vision models)
- **Albumentations Team** - For a fast and flexible augmentation library
- **Hugging Face** - For inspiration on package structure and documentation

### Special Thanks:

- **Flickr8k dataset creators** - For making image-caption data publicly available
- **Open source community** - For countless tutorials, blog posts, and helpful visualizations

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You're free to:
- Use commercially
- Modify
- Distribute
- Use privately

Just include the original license and copyright notice!

---

## 📬 Contact & Connect

- **GitHub**: [Akhan521](https://github.com/Akhan521)
- **LinkedIn**: [Aamir Khan](https://www.linkedin.com/in/aamir-khan-aak521/)
- **Email**: aamirksfg@gmail.com
- **Portfolio**: [aamir-khans-portfolio.vercel.app/](https://aamir-khans-portfolio.vercel.app/)

<br>
<div align="center">

**If you found this project helpful, please consider starring my repository!**


[🔝 Back to Top](#snaption--ai-powered-image-captioning)

</div>
