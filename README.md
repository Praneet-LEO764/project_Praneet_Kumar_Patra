# Face Mask Detection Project

This repository contains code for a face mask detection model using Convolutional Neural Networks (CNNs).

## Dataset

⚠️ **Note:**  
The `data/` folder included in this repository is **not** the complete dataset used during model training.  
The full dataset with over **7,500 images** can be found here:

> [Face Mask Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

You can download it and place it appropriately if you want to retrain or test the model more extensively.

## Model Architecture

- **3 Convolutional Layers (CNN)**
- **2 Fully Connected (Dense) Layers**
- Final output layer for **binary classification** (Mask / No Mask)

## Image Processing

- All input images are resized to **224 × 224 pixels** before being fed into the model.

## Project Structure

```plaintext
Project/
├── model.py        # Defines the CNN architecture
├── train.py        # Handles training loop
├── predict.py      # Inference and prediction code
├── dataset.py      # Custom dataset and dataloader definitions
├── config.py       # Configuration settings (batch size, epochs, etc.)
├── interface.py    # Standardized imports for consistency
├── test.py         # Script to test the model on new data
├── checkpoints/    # Folder containing trained model weights
└── data/           # Dataset folder (partial or full)
