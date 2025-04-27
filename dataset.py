"""
Dataset management for the unicorn image classifier.
Contains custom dataset class and dataloader function.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from PIL import Image
from config import (
    RESIZE_HEIGHT, RESIZE_WIDTH, BATCH_SIZE, 
    NUM_WORKERS, VALIDATION_SPLIT
)

class UnicornImageDataset(torchvision.datasets.ImageFolder):
    """
    Custom dataset class for unicorn image classification.
    Extends ImageFolder to handle custom preprocessing.
    """
    def __init__(self, root, transform=None):
        super(UnicornImageDataset, self).__init__(root=root, transform=transform)
        self.class_names = self.classes  # Store class names for reference
        
    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a random placeholder image in case of errors
            placeholder = torch.randn(3, RESIZE_HEIGHT, RESIZE_WIDTH)
            return placeholder, label
    
    def get_class_names(self):
        """Return the class names for the dataset"""
        return self.class_names


def get_data_transforms():
    """
    Returns the data transformations for training and validation.
    Training includes data augmentation, validation doesn't.
    """
    # Training transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Validation transformations (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir):
    """
    Creates train and validation data loaders from the given directory.
    """
    train_transform, val_transform = get_data_transforms()
    
    # Create full dataset
    full_dataset = UnicornImageDataset(root=data_dir, transform=train_transform)
    
    # Split into train and validation
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms to each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.get_class_names()
