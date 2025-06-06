import os
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets

import kagglehub

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def load_data(domain="art_painting",  split='train', batch_size=32, image_size=224, val_split=0.2, seed=0):
    # Download and locate PACS dataset
    dataset_path = kagglehub.dataset_download("nickfratto/pacs-dataset")
    data_dir = os.path.join(dataset_path, "pacs_data", "pacs_data", domain)

    # Define transforms
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    if split == 'train':
        train_dataset, val_dataset = train_val_dataset(full_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
