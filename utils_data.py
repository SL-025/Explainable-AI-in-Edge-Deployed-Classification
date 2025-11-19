import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transforms():
    """Data transforms for training and evaluation."""
    transform_train = transforms.Compose([
        transforms.Resize(224),                 # CIFAR (32x32) -> 224x224
        transforms.RandomHorizontalFlip(0.5),  # light augmentation
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform_train, transform_eval

def make_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: int = 5000,
    seed: int = 42
):
    """Downloads CIFAR-10 if missing. Uses ALL 10 CLASSES (no filtering).
    Splits the 50k train into train/val (45k/5k by default).
    Returns train_loader, val_loader, test_loader.
    """
    g = torch.Generator().manual_seed(seed)
    t_train, t_eval = make_transforms()

    #FULL CIFAR-10 — no class subset/filtering here
    train_full = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=t_train)
    test_set   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=t_eval)

    train_len = len(train_full) - val_split
    train_set, val_set = random_split(train_full, [train_len, val_split], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
