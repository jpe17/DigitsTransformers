"""
Simple Training Loop
==================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    print(f"Training for {epochs} epochs with lr={lr}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, running_correct, running_total = 0, 0, 0
        
        # tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (patches, labels) in enumerate(pbar):
            optimizer.zero_grad()
            logits = model(patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Update running stats
            running_loss += loss.item()
            predicted = logits.argmax(dim=1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            
            # Show first batch tensor flow
            if epoch == 0 and batch_idx == 0:
                print("--- First batch tensor flow ---")
            
            # Update progress bar with current stats
            curr_loss = running_loss / (batch_idx + 1)
            curr_acc = 100 * running_correct / running_total
            pbar.set_postfix({'Loss': f'{curr_loss:.4f}', 'Acc': f'{curr_acc:.1f}%'})
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for patches, labels in val_loader:
                logits = model(patches)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                predicted = logits.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Final epoch stats
        final_train_acc = 100 * running_correct / running_total
        final_val_acc = 100 * val_correct / val_total
        final_train_loss = running_loss / len(train_loader)
        final_val_loss = val_loss / len(val_loader)
        
        print(f"→ Train: {final_train_loss:.4f} ({final_train_acc:.1f}%) | "
              f"Val: {final_val_loss:.4f} ({final_val_acc:.1f}%)")
    
    print("\n🎉 Training complete!") 