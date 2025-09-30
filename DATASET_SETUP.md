# 📁 Flickr8k Dataset Setup Guide

Quick guide to download and set up the Flickr8k dataset for custom training with Snaption.

> **Note**: The dataset is only needed if you want to **train your own models**. If you're using our [pre-trained models](https://github.com/Akhan521/Snaption/releases/tag/v0.1.0), you can skip this entirely!

## Quick Setup (5 minutes)

1. **Get Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com) (free)

2. **Download Dataset**: Go to [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - Click the **"Download"** button
   - This downloads `archive.zip` (~1GB)

3. **Extract and Organize**:
   - Create an empty directory named `Flickr8k` within `training/data`. The path should be `training/data/Flickr8k`.
   - Extract the downloaded `archive.zip` file's contents into this newly created `Flickr8k` directory.
   - After extraction, the structure should look like this:
     ```
     training/data/Flickr8k/
     ├── captions.txt
     └── Images/
         ├── image1.jpg
         └── ... (8,090 more images)
     ```

4. **Verify**:
   Ensure `captions.txt` and the `Images/` folder with `.jpg` files are present in `training/data/Flickr8k/`.

5. **Note**: 
    Please delete the first line of `captions.txt` if it contains any header information.

## What's Next?

### Option A: Use Pre-Trained Models (Recommended)
```bash
# Download our trained models instead...
# See: https://github.com/Akhan521/Snaption/releases/tag/v0.1.0

# Quick start with pre-trained model:
pip install -e . # Install snaption package
```

### Option B: Train Your Own Model
```bash
# Start training with the dataset you just downloaded:
cd training/
python train.py
```

## 🔧 Troubleshooting

**"Download failed" or "File not found"**
- Make sure you're logged into Kaggle
- Try downloading manually from the website
- Check your internet connection

**"No such file or directory"**
- Make sure you extracted the zip file
- Check that the path `training/data/Flickr8k/` exists
- Verify the Images/ folder contains .jpg files

**"Not enough space"**  
- The dataset needs ~1GB free space.

## 📖 Dataset Info

- **Total Size**: ~1GB
- **Images**: 8,091 photos from Flickr8k
- **Captions**: 5 captions per image (total 40,455 captions)
- **License**: Research/academic use
- **Quality**: Professional photography, diverse scenes

## 🤔 FAQ

**Q: Do I need the dataset to use Snaption?**  
A: No! Only if you want to train your own models. Use our pre-trained models instead.

**Q: Can I use a different dataset?**  
A: Yes! Just follow the same `captions.txt` format. See `training/dataset.py` for details.

**Q: Is this dataset free?**  
A: Yes, it's available for research and educational use.

---

Need help? [Open an issue](https://github.com/Akhan521/Snaption/issues) and I'll assist you! 