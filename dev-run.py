from loader import MnistLoader


def dev_run():
    mnist_loader = MnistLoader()
    train_loader, validation_loader = mnist_loader.get_loaders()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}: Data shape: {data.shape}, Target shape: {target.shape}")
        if batch_idx == 2:  # Limit to first 3 batches for demonstration
            break
        
        
if __name__ == "__main__":
    dev_run()