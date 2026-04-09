# finetune_model.py

import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import load_datasets
from models.resnet50_attention import ResNet50_NonLocal


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
    # Initialize Model
    # ---------------------------------
    model = ResNet50_NonLocal(num_classes=2)


    # ---------------------------------
    # Load SSL Pretrained Encoder
    # ---------------------------------
    try:
        ssl_weights = torch.load("simclr_model.pth", map_location=device)
        
        # Map SimCLR encoder sequential keys to ResNet backbone keys
        key_mapping = {
            'encoder.0.': 'conv1.',
            'encoder.1.': 'bn1.',
            'encoder.4.': 'layer1.',
            'encoder.5.': 'layer2.',
            'encoder.6.': 'layer3.',
            'encoder.7.': 'layer4.',
            'encoder.8.': 'avgpool.'
        }
        
        mapped_weights = {}
        for k, v in ssl_weights.items():
            for prefix, new_prefix in key_mapping.items():
                if k.startswith(prefix):
                    mapped_weights[k.replace(prefix, new_prefix, 1)] = v
                    break
                    
        # strict=False allows our non-local attention weights to remain randomly initialized
        model.load_state_dict(mapped_weights, strict=False)
        print("Loaded SSL pretrained weights from simclr_model.pth.")
    except FileNotFoundError:
        print("simclr_model.pth not found. EXITING!!")
        exit()
    except Exception as e:
        print(f"Error loading SSL weights: {e}")


    model = model.to(device)


    # ---------------------------------
    # Loss and Optimizer
    # ---------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )


    # ---------------------------------
    # Training Settings
    # ---------------------------------
    epochs = 5 #TESTING WITH 5 then rolling over to 50? idk


    # ---------------------------------
    # Training Loop
    # ---------------------------------
    for epoch in range(epochs):
        print(f"Training Epoch: {epoch+1}...")
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
        model.eval()
        print(f"Validating Epoch: {epoch+1}...")
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
    # Save Fine-Tuned Model
    # ---------------------------------
    torch.save(model.state_dict(), "fruit_freshness_model.pth")
    print("Model saved as fruit_freshness_model.pth")

if __name__ == '__main__':
    main()