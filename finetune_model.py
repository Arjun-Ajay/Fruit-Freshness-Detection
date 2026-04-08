# finetune_model.py

import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import load_datasets
from models.resnet_attention import ResNet101_NonLocal


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
model = ResNet101_NonLocal(num_classes=2)


# ---------------------------------
# Load SSL Pretrained Encoder
# ---------------------------------
try:
    encoder_weights = torch.load("ssl_encoder.pth", map_location=device)
    model.layer1.load_state_dict(encoder_weights, strict=False)
    print("Loaded SSL pretrained weights.")
except:
    print("SSL weights not found. Training from ImageNet weights.")


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
epochs = 30


# ---------------------------------
# Training Loop
# ---------------------------------
for epoch in range(epochs):

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