from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class MudraDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))

        for i, cls in enumerate(self.classes):
            folder = os.path.join(root_dir, cls)
            for img in os.listdir(folder):
                self.samples.append((os.path.join(folder, img), i))

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
