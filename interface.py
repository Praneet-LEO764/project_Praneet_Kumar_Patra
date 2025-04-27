"""
Main script to train the unicorn image classifier and make predictions.
"""

import os
import argparse
import torch
import random
import numpy as np
from model import create_model
from dataset import create_data_loaders
from train import train_model, evaluate_model
from predict import predict_batch, visualize_predictions

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Unicorn Image Classifier')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='Operation mode: train or predict')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for prediction or resuming training')
    parser.add_argument('--predict_dir', type=str, default=None,
                        help='Directory containing images for prediction')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Create data loaders
        train_loader, val_loader, class_names = create_data_loaders(args.data_dir)
        print(f"Classes: {class_names}")
        
        # Create model
        model = create_model()
        model = model.to(device)
        
        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model checkpoint from {args.checkpoint}")
        
        # Train model
        trained_model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            device,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints')
        )
        
        # Evaluate on validation set
        print("Evaluating model on validation set...")
        evaluate_model(trained_model, val_loader, device)
        
    elif args.mode == 'predict':
        # Load model from checkpoint
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            print("Error: Checkpoint required for prediction mode")
            return
        
        # Get class names from data directory
        _, _, class_names = create_data_loaders(args.data_dir)
        print(f"Classes: {class_names}")
        
        # Create model and load checkpoint
        model = create_model()
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"Loaded model checkpoint from {args.checkpoint}")
        
        # Get image paths for prediction
        if not args.predict_dir or not os.path.exists(args.predict_dir):
            print("Error: Prediction directory required")
            return
        
        image_paths = [
            os.path.join(args.predict_dir, f)
            for f in os.listdir(args.predict_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not image_paths:
            print("No images found in the prediction directory")
            return
        
        print(f"Making predictions on {len(image_paths)} images...")
        predictions = predict_batch(model, image_paths, device, class_names)
        
        # Visualize predictions
        output_dir = os.path.join(args.output_dir, 'predictions')
        visualize_predictions(image_paths, predictions, class_names, output_dir)
        print(f"Saved prediction visualizations to {output_dir}")
        
        # Print prediction results
        for pred in predictions:
            print(f"Image: {os.path.basename(pred['image_path'])}, "
                  f"Predicted: {pred['predicted_class']}, "
                  f"Confidence: {pred['confidence']:.4f}")

if __name__ == '__main__':
    main()
