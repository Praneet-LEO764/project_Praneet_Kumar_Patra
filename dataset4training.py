import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from config import resize_x, resize_y, batchsize

class UnicornImgDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def unicornLoader(root_dir):
    dataset = UnicornImgDataset(root=root_dir, transform=transform1)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False)
    return loader



