# test.py

import torch
from interface import TheModel, the_dataloader

def main():
    # Load test dataloader
    test_loader = the_dataloader('./data')

    # Load model
    model = TheModel()

    # If you have a saved model checkpoint, load it here
    model.load_state_dict(torch.load('./checkpoints/final_weights.pth'))
    # Uncomment below if you have a model checkpoint
    # model.load_state_dict(torch.load('path_to_saved_model.pth'))

    model.eval()  # Set model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
