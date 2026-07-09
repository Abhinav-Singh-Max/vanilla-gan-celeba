# Vanilla GAN — CelebA Face Generation

A from-scratch implementation of a **Vanilla GAN (Generative Adversarial Network)** using fully-connected (dense) layers, trained on the **CelebA** dataset to generate 64x64 human face images.

## Overview

This project implements the original GAN architecture (Goodfellow et al., 2014) — no convolutions, just linear layers — as a foundational exercise in understanding adversarial training before moving to DCGAN/StyleGAN-style architectures.

## Architecture

**Generator**
- Input: 100-dim random noise vector
- 4 fully-connected layers (100 → 256 → 512 → 1024 → 12288)
- ReLU activations, Tanh output (scaled to [-1, 1])
- Reshaped to 3x64x64 RGB image

**Discriminator**
- Flattened 3x64x64 image → 4 fully-connected layers (12288 → 1024 → 512 → 256 → 1)
- LeakyReLU activations, Sigmoid output (real/fake probability)

## Dataset

- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — 202,599 aligned & cropped celebrity face images
- Preprocessing: center-crop (178x178) → resize (64x64) → normalize to [-1, 1]

## Training Details

- Loss: Binary Cross-Entropy (BCE)
- Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999)) for both Generator and Discriminator
- Batch size: 64
- Trained on GPU (CUDA / Apple MPS support included, CPU fallback)

## Results

Generated samples are saved after every epoch to visually track how the Generator improves over training. Sample loss curve (epoch 1):

| Batch | G Loss | D Loss |
|-------|--------|--------|
| 1     | 0.68   | 0.70   |
| 500   | 2.61   | 0.37   |
| 1000  | 3.30   | 0.07   |
| 1600  | 2.39   | 0.22   |

## How to Run

1. Download the [CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
2. Update `root_dir_path` to point to your dataset folder
3. Run all cells in `Vanilla_gan_main.ipynb` sequentially

```bash
pip install torch torchvision matplotlib pillow numpy
```

## Key Learnings

- Building a GAN training loop from scratch (alternating D/G updates)
- Why fully-connected GANs struggle with image structure compared to convolutional GANs
- Debugging adversarial training instability (loss oscillation between G and D)

## Future Improvements

- Replace dense layers with convolutional layers (DCGAN)
- Add batch normalization for training stability
- Experiment with label smoothing / different loss functions (WGAN)

---

*Built as part of my deep learning journey into Generative Adversarial Networks.*
