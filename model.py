"""
Deep learning model architecture for unicorn image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, NUM_CLASSES

class UnicornCNN(nn.Module):
    """
    CNN architecture for unicorn image classification.
    Improved architecture with batch normalization, dropout, and skip connections.
    """
    def __init__(self):
        super(UnicornCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the feature maps before the fully connected layer
        # After 4 max pooling layers with stride 2, the feature map size is reduced by 2^4 = 16
        feature_size = 128 * (224 // 16) * (224 // 16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, NUM_CLASSES)
        
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


def create_model():
    """
    Factory function to create and initialize the model.
    """
    model = UnicornCNN()
    
    # Initialize weights using He initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    return model
