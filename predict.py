import torch
from PIL import Image
from torchvision import transforms
from config import resize_x, resize_y

def cryptic_inf_f(model, list_of_img_paths, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    images = []
    for img_path in list_of_img_paths:
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        images.append(image)

    batch = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(batch)
        _, predicted = torch.max(outputs, 1)

    return predicted.cpu().tolist()
