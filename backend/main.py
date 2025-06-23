"""
Simple Vision Transformer Demo
============================
"""

import torch
import os
from data_processing import setup_data_loaders
from transformer_architecture import VisionTransformerEncoder
from training_engine import train_model


def main():
    print("=== Simple Vision Transformer Demo ===\n")
    
    # Create artifacts directory if it doesn't exist
    # Get the project root directory (parent of backend)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader = setup_data_loaders(batch_size=32)
    print()
    
    # Create model
    model = VisionTransformerEncoder(
        patch_dim=49,      # 7x7 = 49
        embed_dim=32,      # Small embedding dimension
        num_patches=16,    # 4x4 grid of patches
        num_classes=10,    # MNIST digits 0-9
        num_heads=2,       # Multi-head attention
        num_layers=2       # Just 2 transformer layers
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Show tensor flow on first batch
    print("=== Tensor Flow Demo ===")
    sample_patches, sample_labels = next(iter(train_loader))
    print(f"Processing batch with {len(sample_labels)} images")
    
    with torch.no_grad():
        logits = model(sample_patches)
        predictions = logits.argmax(dim=1)
        
    print(f"\nPredictions: {predictions[:8].tolist()}")
    print(f"Actual:      {sample_labels[:8].tolist()}")
    print()
    
    # Train the model
    print("=== Training ===")
    train_model(model, train_loader, val_loader, epochs=5, lr=0.001)
    
    # Save the trained model
    model_path = os.path.join(artifacts_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'patch_dim': 49,
        'embed_dim': 32,
        'num_patches': 16,
        'num_classes': 10,
        'num_heads': 2,
        'num_layers': 2,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    metadata_path = os.path.join(artifacts_dir, "model_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("Vision Transformer Model Metadata\n")
        f.write("=" * 35 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📋 Model metadata saved to: {metadata_path}")
    print("\n✅ Training complete! Model artifacts saved.")


if __name__ == "__main__":
    main() 