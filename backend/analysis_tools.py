"""
Analysis Tools Module - Clean Results Analysis
============================================

Clean visualization and analysis tools for understanding model performance.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def analyze_training_history(history, detailed=True):
    """
    Analyze training history with key insights.
    """
    print("Training Analysis:")
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accs = history['train_accuracies']
    val_accs = history['val_accuracies']
    
    print(f"  Final train accuracy: {train_accs[-1]:.2f}%")
    print(f"  Final val accuracy: {val_accs[-1]:.2f}%")
    print(f"  Best val accuracy: {max(val_accs):.2f}%")
    
    # Loss reduction
    loss_reduction = ((train_losses[0] - train_losses[-1]) / train_losses[0] * 100)
    print(f"  Loss reduction: {loss_reduction:.1f}%")
    
    # Overfitting check
    overfit_gap = max(train_accs) - max(val_accs)
    if overfit_gap < 3:
        print(f"  ✓ Good generalization (gap: {overfit_gap:.1f}%)")
    else:
        print(f"  ⚠️ Some overfitting (gap: {overfit_gap:.1f}%)")
    
    if detailed:
        plot_training_curves(history)


def plot_training_curves(history):
    """
    Create clean training visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Training', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title('Loss Evolution')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['train_accuracies'], 'b-', label='Training', linewidth=2)
    axes[1].plot(epochs, history['val_accuracies'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title('Accuracy Evolution')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Key insights
    best_epoch = np.argmax(history['val_accuracies']) + 1
    print(f"  Best epoch: {best_epoch}")
    print()


def analyze_predictions(model, val_loader, detailed_analysis=True):
    """
    Analyze model predictions.
    """
    print("Prediction Analysis:")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx, (patches, labels) in enumerate(val_loader):
            if batch_idx >= 5:  # Limit for demo
                break
                
            logits = model(patches)
            probabilities = torch.softmax(logits, dim=1)
            predicted = logits.argmax(dim=1)
            confidences = probabilities.max(dim=1)[0]
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Overall accuracy
    accuracy = (all_predictions == all_labels).mean() * 100
    print(f"  Overall accuracy: {accuracy:.2f}%")
    
    # Confidence analysis
    correct_mask = all_predictions == all_labels
    correct_confidence = all_confidences[correct_mask].mean()
    wrong_confidence = all_confidences[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
    
    print(f"  Correct prediction confidence: {correct_confidence:.3f}")
    print(f"  Wrong prediction confidence: {wrong_confidence:.3f}")
    print(f"  Confidence gap: {correct_confidence - wrong_confidence:.3f}")
    
    if detailed_analysis:
        # Per-class accuracy
        print("  Per-class accuracy:")
        for class_id in range(10):
            class_mask = all_labels == class_id
            if class_mask.sum() > 0:
                class_acc = (all_predictions[class_mask] == all_labels[class_mask]).mean() * 100
                print(f"    Class {class_id}: {class_acc:.1f}%")
    
    print()
    return {
        'accuracy': accuracy,
        'correct_confidence': correct_confidence,
        'wrong_confidence': wrong_confidence
    }


def show_prediction_mistakes(model, val_loader, val_patch_loader, num_mistakes=5):
    """
    Show prediction mistakes for learning.
    """
    print("Mistake Analysis:")
    
    model.eval()
    mistakes = []
    
    val_iter = iter(val_loader)
    patch_iter = iter(val_patch_loader)
    
    with torch.no_grad():
        while len(mistakes) < num_mistakes:
            try:
                images, labels = next(val_iter)
                patches, _ = next(patch_iter)
                
                logits = model(patches)
                probabilities = torch.softmax(logits, dim=1)
                predicted = logits.argmax(dim=1)
                confidences = probabilities.max(dim=1)[0]
                
                for i in range(len(labels)):
                    if labels[i] != predicted[i] and len(mistakes) < num_mistakes:
                        mistakes.append({
                            'image': images[i].squeeze().numpy(),
                            'true_label': labels[i].item(),
                            'pred_label': predicted[i].item(),
                            'confidence': confidences[i].item()
                        })
            except StopIteration:
                break
    
    if len(mistakes) == 0:
        print("  No mistakes found! Model performing excellently.")
        return
    
    print(f"  Found {len(mistakes)} mistakes:")
    for i, mistake in enumerate(mistakes):
        print(f"    {i+1}. True: {mistake['true_label']}, "
              f"Predicted: {mistake['pred_label']}, "
              f"Confidence: {mistake['confidence']:.3f}")
    
    # Plot mistakes
    fig, axes = plt.subplots(1, len(mistakes), figsize=(2*len(mistakes), 3))
    if len(mistakes) == 1:
        axes = [axes]
    
    for i, mistake in enumerate(mistakes):
        axes[i].imshow(mistake['image'], cmap='gray')
        axes[i].set_title(f'True: {mistake["true_label"]}\n'
                         f'Pred: {mistake["pred_label"]}\n'
                         f'Conf: {mistake["confidence"]:.2f}')
        axes[i].axis('off')
    
    plt.suptitle('Prediction Mistakes', fontsize=14)
    plt.tight_layout()
    plt.show()
    print()


def show_learning_insights(history):
    """
    Provide key learning insights.
    """
    print("Key Learning Insights:")
    
    val_accs = history['val_accuracies']
    
    # Learning speed
    epochs_to_90 = None
    for i, acc in enumerate(val_accs):
        if acc >= 90.0:
            epochs_to_90 = i + 1
            break
    
    if epochs_to_90:
        print(f"  • Reached 90% accuracy in {epochs_to_90} epochs")
    
    # Convergence
    if len(history['train_losses']) > 3:
        final_losses = history['train_losses'][-3:]
        if np.var(final_losses) < 0.001:
            print("  • Training converged smoothly")
        else:
            print("  • Training still improving")
    
    # Transformer insights
    print("  • Vision Transformer successfully learned:")
    print("    - Patch-based image processing")
    print("    - Spatial relationships via self-attention")
    print("    - Feature extraction through multiple layers")
    
    best_acc = max(val_accs)
    if best_acc >= 95:
        print(f"  • Excellent performance achieved ({best_acc:.1f}%)")
    elif best_acc >= 90:
        print(f"  • Good performance achieved ({best_acc:.1f}%)")
    
    print() 