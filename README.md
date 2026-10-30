# CIFAR-10 Image Classification with a Dense Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YISHU19/Deep-Learning-Model/blob/main/DL2.ipynb)

A baseline deep learning model that classifies images from the **CIFAR-10** dataset (60,000 32×32 colour images across 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

## Model
A fully connected (dense) network built with **TensorFlow / Keras**:

| Layer | Details |
|---|---|
| Flatten | 32 × 32 × 3 input → 3,072 values |
| Dense | 512 units, ReLU |
| Dense | 256 units, ReLU |
| Dense | 128 units, ReLU |
| Dense | 10 units, Softmax |

- **Optimizer:** Adam
- **Loss:** sparse categorical cross-entropy
- **Training:** 10 epochs, batch size 128

## Results
Training accuracy reached about **40%** after 10 epochs.

## Next steps
- Train and evaluate on the normalised data (`x_train`, `x_test`) instead of the raw pixel values — the notebook currently evaluates on unscaled images, which drags test accuracy down
- Replace the dense layers with convolutional layers (a CNN), which suit image data much better
- Add data augmentation and plot training/validation curves

## How to run
Open the notebook in Google Colab using the badge above — CIFAR-10 downloads automatically, no setup needed.

## Tech
Python · TensorFlow · Keras · NumPy · Matplotlib · OpenCV · Pillow
