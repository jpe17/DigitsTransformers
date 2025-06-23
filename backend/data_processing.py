"""
Data Processing and Loading
=========================
"""

import torch
from torch.utils.data import random_split, DataLoader
from torch import reshape
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MnistLoader():
    def __init__(self, datadir='.w3-data/', batch_size=128):
        self.mnist_dataset = MNIST(root=datadir, download=True, train=True, transform=transforms.ToTensor())
        self.train_data, self.validation_data = random_split(self.mnist_dataset, [50000, 10000])
        self.train_loader = DataLoader(self.train_data, batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_data, batch_size, shuffle=False)
        print(f"mnist dataset loaded, train data size: {len(self.train_data)} validation data size: {len(self.validation_data)}")
        
    def get_loaders(self):
        return self.train_loader, self.validation_loader


class SquareImageSplitingLoader():
    def __init__(self, loader: DataLoader, number_of_segments=16, segment_dimension=7):
        self.loader = loader
        self.number_of_segments = number_of_segments
        self.segment_dimension = segment_dimension
    
    def __len__(self):
        """Return the length of the underlying loader"""
        return len(self.loader)
    
    def __iter__(self):
        kc, kh, kw = 1, self.segment_dimension, self.segment_dimension  # kernel size
        dc, dh, dw = 1, self.segment_dimension, self.segment_dimension  # stride
    
        for index, batch in enumerate(self.loader):
            batch[0] = batch[0].unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
            batch[0] = batch[0].contiguous().view(-1, kc, kh, kw)
            batch[0] = reshape(batch[0], (-1, self.number_of_segments, 1, self.segment_dimension, self.segment_dimension))
            yield batch


def setup_data_loaders(batch_size=32):
    print("Loading MNIST data...")
    
    # Load MNIST data
    mnist_loader = MnistLoader(batch_size=batch_size)
    train_loader, val_loader = mnist_loader.get_loaders()
    
    # Create patch loaders
    train_patch_loader = SquareImageSplitingLoader(train_loader)
    val_patch_loader = SquareImageSplitingLoader(val_loader)
    
    # Show data transformation
    sample_patches, sample_labels = next(iter(train_patch_loader))
    sample_images, _ = next(iter(train_loader))
    
    print(f"Original images: {sample_images.shape}")
    print(f"Patch format: {sample_patches.shape}")
    print(f"Transformation: 28x28 image → 16 patches of 7x7")
    
    return train_patch_loader, val_patch_loader 