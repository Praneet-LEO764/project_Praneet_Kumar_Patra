# Face mask detection

This project implements a deep learning-based binary image classifier using PyTorch, designed to identify unicorn images. The classifier uses a convolutional neural network (CNN) architecture and includes training, evaluation, and prediction functionality.

## Project Structure

```
Face-mask-detection/
├── config.py           # Configuration parameters
├── dataset.py          # Dataset loading and preprocessing
├── model.py            # Neural network architecture
├── train.py            # Training and evaluation functions
├── predict.py          # Functions for making predictions
├── main.py             # Main script for running the model
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```

## Key Features

- **Data augmentation**: Random flips , rotations upto 40 degrees, a bit of gaussian blurr
- **Model architecture**: Custom CNN with batch normalization , 3 cnn layers and 2 dense neural networks
- **Evaluation**: Detailed metrics including accuracy, precision, recall, and F1 score
- **Prediction**: Easy-to-use functions for making predictions on new images

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/unicorn-classifier.git
   cd unicorn-classifier
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:

```
python interface.py --mode train --data_dir /path/to/data --output_dir results
```

The data directory should have the following structure:
```
data/
├── with_mask/          
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── without_mask/      
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### Making Predictions

To make predictions on new images:

```
python interface.py --mode predict --data_dir /path/to/data --checkpoint results/checkpoints/best_model.pth --predict_dir /path/to/test_images --output_dir results
```

## Customization

You can customize the model by modifying the following files:

- `config.py`: Adjust hyperparameters such as batch size, learning rate, and image dimensions
- `model.py`: Modify the neural network architecture
- `dataset.py`: Change the data preprocessing and augmentation steps

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- Pillow
- numpy
- matplotlib
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
