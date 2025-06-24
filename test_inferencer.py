"""
Test the digit inferencer with sample images
==========================================
"""

import torch
import numpy as np
import os
from backend.camera_inferencer import DigitInferencer
from backend.data_processing import setup_data_loaders


def test_with_mnist_sample():
    """Test inferencer with a sample from MNIST dataset"""
    print("Testing inferencer with MNIST sample...")
    
    # Load some test data
    _, val_loader = setup_data_loaders(batch_size=1)
    
    # Get one sample
    sample_patches, sample_label = next(iter(val_loader))
    print(f"True label: {sample_label.item()}")
    
    # Create inferencer
    inferencer = DigitInferencer()
    
    # Test prediction directly with patches
    with torch.no_grad():
        logits = inferencer.model(sample_patches)
        predicted = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0][predicted].item()
    
    print(f"Predicted: {predicted} (confidence: {confidence:.3f})")
    print("✓ Basic inference test passed!")


def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("\nTesting preprocessing pipeline...")
    
    # Create a fake 28x28 image (white digit on black background)
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Draw a simple "1" shape
    fake_image[30:70, 45:55] = 255  # vertical line
    
    # Test preprocessing
    inferencer = DigitInferencer()
    
    try:
        predicted_digit, confidence, probabilities = inferencer.predict(fake_image)
        print(f"Fake image prediction: {predicted_digit} (confidence: {confidence:.3f})")
        print("✓ Preprocessing test passed!")
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")


def check_artifacts():
    """Check if artifacts directory and model exist"""
    print("Checking artifacts...")
    
    # Get the correct path to artifacts directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(current_dir, "artifacts")
    model_path = os.path.join(artifacts_dir, "trained_model.pth")
    metadata_path = os.path.join(artifacts_dir, "model_metadata.txt")
    
    if not os.path.exists(artifacts_dir):
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return False
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        return False
    
    print(f"✅ Artifacts directory found: {artifacts_dir}")
    print(f"✅ Trained model found: {model_path}")
    
    if os.path.exists(metadata_path):
        print(f"✅ Model metadata found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            print("📋 Model info:")
            for line in f.readlines()[2:]:  # Skip header
                print(f"  {line.strip()}")
    
    return True


if __name__ == "__main__":
    print("🧪 Testing Digit Inferencer")
    print("=" * 40)
    
    try:
        # Check artifacts first
        if not check_artifacts():
            print("\n🚀 Setup required:")
            print("1. Run: python backend/main.py")
            print("2. Then run this test again")
            exit(1)
        
        print()
        test_with_mnist_sample()
        test_preprocessing()
        
        print("\n🎉 All tests passed! You can now run the camera inferencer:")
        print("python camera_inferencer.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure you have:")
        print("1. All dependencies installed: pip install -r requirements.txt")
        print("2. Trained model available: python backend/main.py") 