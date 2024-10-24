import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + ".png")
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_train_loader(config):
    train_transforms = transforms.Compose([
        transforms.Resize((config['hyperparameters']['image_size']['height'],
                           config['hyperparameters']['image_size']['width'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = DiabeticRetinopathyDataset(
        csv_file=config['data']['train_csv'],
        img_dir=config['data']['train_images_dir'],
        transform=train_transforms
    )
    
    loader = DataLoader(dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True)
    return loader
