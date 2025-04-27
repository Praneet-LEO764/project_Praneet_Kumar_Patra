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

transform2 = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def unicornLoader(root_dir):
    dataset = UnicornImgDataset(root=root_dir, transform=transform2)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False)
    return loader
