# train_convnext_tiny.py

import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import load_datasets
from models.convnext_tiny import ConvNeXt_Tiny_Model

def main():
    # ---------------------------------
    # Device Setup
    # ---------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    train_loader, val_loader, _ = load_datasets()

    # ---------------------------------
    # Initialize ConvNeXt Tiny
    # ---------------------------------
    print("Loading Custom ConvNeXt Tiny Architecture...")
    model = ConvNeXt_Tiny_Model(num_classes=2)
    model = model.to(device)

    # ---------------------------------
    # Loss and Optimizer
    # ---------------------------------
    criterion = nn.CrossEntropyLoss()
    
    # We are training from absolute scratch, so optimize all parameters
    optimizer = optim.Adam(
        model.parameters(), 
        lr=1e-4
    )

    # ---------------------------------
    # Training Settings
    # ---------------------------------
    epochs = 55
    best_val_acc = 0.0

    # ---------------------------------
    # Training Loop
    # ---------------------------------
    for epoch in range(epochs):
        print(f"\nTraining Epoch: {epoch+1}...")
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100 * correct / total

        # --------------------------
        # Validation
        # --------------------------
        print(f"Validating Epoch: {epoch+1}...")
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss: {running_loss/len(train_loader):.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Acc: {val_accuracy:.2f}%"
        )
        
        # ---------------------------------
        # AUTOSAVE FEATURE (Early Stopping Protection)
        # ---------------------------------
        # Instead of saving blindly at the end, we only save if the model breaks a new high score
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "convnext_tiny_model.pth")
            print(f"⭐ New Personal Best! Model saved as 'convnext_tiny_model.pth' with {val_accuracy:.2f}% accuracy.")

    print(f"\nTraining Complete! Best Validation Accuracy achieved: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
