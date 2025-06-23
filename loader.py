from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class MnistLoader():
    def __init__(self, datadir = '39-data/', batch_size = 128):
        self.mnist_dataset = MNIST(root = datadir, download=True, train = True, transform = transforms.ToTensor())
        self.train_data, self.validation_data = random_split(self.mnist_dataset, [50000, 10000])
        self.train_loader = DataLoader(self.train_data, batch_size, shuffle = True)
        self.validation_loader = DataLoader(self.validation_data, batch_size, shuffle = False)
        print(f"mnist dataset loaded, train data size: {len(self.train_data)} validation data size: {len(self.validation_data)}")
        
    def get_loaders(self):
        return self.train_loader, self.validation_loader
        