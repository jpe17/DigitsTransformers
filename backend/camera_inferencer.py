"""
Real-time Handwritten Digit Recognition
======================================
"""

import cv2
import torch
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
from transformer_architecture import VisionTransformer
from data_processing import SquareImageSplitingLoader
from data_processing import MnistLoader
from data_processing import setup_data_loaders


class DigitInferencer:
    def __init__(self, model_path=None):
        if model_path is None:
            # Get the correct path to artifacts directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "../artifacts", "trained_model.pth")
        # Initialize model with the same architecture used in training
        self.model = VisionTransformer(
            patch_dim=49,
            embed_dim=32, 
            num_patches=16,
            num_classes=10,
            num_heads=2,
            num_layers=2
        )
        
        # Load trained model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print(f"❌ Model not found at {model_path}")
            print("Please run 'python backend/main.py' first to train and save a model.")
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        self.model.eval()
    
    def preprocess_frame(self, frame):
        """Convert camera frame to MNIST-like format"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Invert colors (MNIST has white digits on black background)
        inverted = 255 - resized
        
        # Normalize to [0, 1]
        normalized = inverted.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
        
        return tensor
    
    def tensor_to_patches(self, image_tensor):
        """Convert 28x28 image tensor to patches like the training data"""
        # Create a dummy DataLoader-like structure
        batch = [image_tensor, torch.tensor([0])]  # dummy label
        
        # Use the same splitting logic as training
        splitter = SquareImageSplitingLoader(iter([batch]))
        patches, _ = next(iter(splitter))
        
        return patches
    
    def predict(self, frame):
        """Predict digit from camera frame"""
        # Preprocess frame
        image_tensor = self.preprocess_frame(frame)
        
        # Convert to patches
        patches = self.tensor_to_patches(image_tensor)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(patches)
            probabilities = F.softmax(logits, dim=1)
            predicted_digit = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
        
        return predicted_digit, confidence, probabilities[0].numpy()
    
    def run_camera_inference(self):
        """Main camera loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("🎥 Starting camera inference...")
        print("Instructions:")
        print("- Show a handwritten digit to the camera")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        
        frame_count = 0
        predicted_digit, confidence, probabilities = -1, 0.0, np.zeros(10)
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Create a region of interest (center square)
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 2
            roi_x = (w - roi_size) // 2
            roi_y = (h - roi_size) // 2
            
            # Extract ROI
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Run prediction every 10 frames to reduce computation
            if frame_count % 10 == 0:
                try:
                    predicted_digit, confidence, probabilities = self.predict(roi)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_digit, confidence = -1, 0.0
                    probabilities = np.zeros(10)
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)
            
            # Draw prediction info
            if predicted_digit >= 0:
                text = f"Digit: {predicted_digit} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw probability bars
                for i in range(10):
                    bar_height = int(probabilities[i] * 200)
                    cv2.rectangle(frame, (10 + i*60, 450), (60 + i*60, 450-bar_height), (255, 0, 0), -1)
                    cv2.putText(frame, str(i), (25 + i*60, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Instructions
            cv2.putText(frame, "Show digit in green box", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Digit Recognition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'captured_frame_{frame_count}.jpg', frame)
                print(f"📸 Frame saved as captured_frame_{frame_count}.jpg")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Camera inference stopped.")


def main():
    try:
        # Create inferencer (will load from artifacts)
        inferencer = DigitInferencer()
        
        # Run camera inference
        inferencer.run_camera_inference()
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n🚀 Quick setup:")
        print("1. Run: python backend/main.py")
        print("2. Then run: python camera_inferencer.py")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 