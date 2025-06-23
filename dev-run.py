from loaders import MnistLoader
from loaders import SquareImageSplitingLoader

def dev_run():
    mnist_loader = MnistLoader()
    train_loader, validation_loader = mnist_loader.get_loaders()
    
    for batch_idx, (data, target) in enumerate(SquareImageSplitingLoader(train_loader)):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        break
        
if __name__ == "__main__":
    dev_run()