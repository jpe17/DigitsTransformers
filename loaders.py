from torch.utils.data import random_split
from torch import reshape
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class MnistLoader():
    def __init__(self, datadir = '.w3-data/', batch_size = 128):
        self.mnist_dataset = MNIST(root = datadir, download=True, train = True, transform = transforms.ToTensor())
        self.train_data, self.validation_data = random_split(self.mnist_dataset, [50000, 10000])
        self.train_loader = DataLoader(self.train_data, batch_size, shuffle = True)
        self.validation_loader = DataLoader(self.validation_data, batch_size, shuffle = False)
        print(f"mnist dataset loaded, train data size: {len(self.train_data)} validation data size: {len(self.validation_data)}")
        
    def get_loaders(self):
        return self.train_loader, self.validation_loader
        
        
class SquareImageSplitingLoader():
    def __init__(self, loader : DataLoader, number_of_segments = 16, segment_dimension = 7):
        self.loader = loader
        self.number_of_segments = number_of_segments
        self.segment_dimension = segment_dimension
    
    def __iter__(self):
        kc, kh, kw = 1, self.segment_dimension, self.segment_dimension  # kernel size
        dc, dh, dw = 1, self.segment_dimension, self.segment_dimension  # stride
    
        for index, batch in enumerate(self.loader):
            batch[0] = batch[0].unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
            batch[0] = batch[0].contiguous().view(-1, kc, kh, kw)
            batch[0] = reshape(batch[0], (-1, self.number_of_segments, 1, self.segment_dimension, self.segment_dimension))
            yield batch
    
    
def test_split():
    dataset = MnistLoader()
    splitting_loader = SquareImageSplitingLoader(dataset.get_loaders()[0]) 
    for idx, batch in enumerate(splitting_loader):
        print(f"BATCH {idx}, batch shape: {batch[0].shape}, batch label shape: {batch[1].shape}")
        for image_index, splitted_image in enumerate(batch[0]):
            for segment_index, segment in enumerate(splitted_image):
                # save_image(segment, f'./test/img{image_index}-segment{segment_index}.png')
                pass
        
if __name__ == "__main__":
    test_split()
    
    from PIL import Image
    images = [Image.open(f'./test/segment{i}.png') for i in range(16)]
    widths, heights = zip(*(i.size for i in images))
    new_im = Image.new('L', (28,28))
    x_offset = 0
    for x in range(4):
        y_offset = 0
        for y in range(4):
            new_im.paste(images[y * 4 + x], (x_offset, y_offset))
            y_offset += 7
        x_offset += 7
    new_im.save('./test/reconstructed.png')