# CIFAR-10 Image Classification: Dense Network vs CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YISHU19/Deep-Learning-Model/blob/main/DL2.ipynb)

Classifying images from the **CIFAR-10** dataset (60,000 32×32 colour images across 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck), comparing a fully connected baseline with a convolutional neural network.

## Results

| Model | Test accuracy |
|---|---|
| Dense neural network (baseline) | **47.0%** |
| Convolutional neural network | **81.0%** |

The CNN wins by a wide margin because convolutional layers learn local patterns (edges, textures, shapes) and keep the spatial layout of the image, which a dense network throws away when it flattens the pixels.

## Models

**Dense baseline:** Flatten → Dense 512 → Dense 256 → Dense 128 → Dense 10 (softmax), trained for 10 epochs.

**CNN:** three convolutional blocks (32 → 64 → 128 filters), each with two 3×3 convolutions, batch normalisation, max pooling and dropout, followed by a 256-unit dense layer. Random horizontal flips are used as data augmentation. Trained for 15 epochs.

Both use the Adam optimiser, sparse categorical cross-entropy loss, batch size 128 and a 10% validation split.

## What I fixed

An earlier version trained and evaluated on raw 0–255 pixel values while the normalised arrays went unused, so test accuracy collapsed to 10% (random guessing). The notebook now scales pixels to 0–1 and uses the same data for training and evaluation.

## How to run

Open the notebook in Google Colab with the badge above. CIFAR-10 downloads automatically and no setup is needed. A GPU runtime makes training much faster.

The notebook uses standalone Keras 3, so it runs on any Keras backend (TensorFlow, PyTorch or JAX).

## Tech

Python · Keras · NumPy · Matplotlib
