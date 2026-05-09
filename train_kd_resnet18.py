import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from preprocessing import load_datasets
from models.resnet50_attention import ResNet50_NonLocal

def get_student_model(num_classes=2):
    model = models.resnet18(weights=None)
    # Change the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def loss_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature T and weight alpha.
    """
    # Hard loss (standard cross entropy)
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    
    # Soft loss (KL Divergence between soft labels)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1)
    ) * (alpha * T * T)
    
    return hard_loss + soft_loss

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
    # Initialize Models
    # ---------------------------------
    # 1. Teacher Model
    print("Loading Teacher Model (ResNet50_NonLocal)...")
    teacher_model = ResNet50_NonLocal(num_classes=2)
    try:
        teacher_model.load_state_dict(torch.load("fruit_freshness_model.pth", map_location=device))
        print("Successfully loaded teacher weights.")
    except Exception as e:
        print(f"Error loading teacher weights: {e}")
        return
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval() # Teacher is always in eval mode

    # 2. Student Model
    print("Initializing Student Model (ResNet18)...")
    student_model = get_student_model(num_classes=2)
    student_model = student_model.to(device)

    # ---------------------------------
    # Optimizer and KD Params
    # ---------------------------------
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    
    epochs = 15 # Default epoch count for testing the KD logic
    alpha = 0.5 # Weight for KD Loss vs CE Loss
    T = 3.0     # Temperature

    # ---------------------------------
    # Training Loop
    # ---------------------------------
    for epoch in range(epochs):
        print(f"\nTraining Epoch: {epoch+1}/{epochs}...")
        student_model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Forward pass Teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            # 2. Forward pass Student
            student_outputs = student_model(images)

            # 3. Calculate KD Loss
            loss = loss_kd(student_outputs, labels, teacher_outputs, alpha, T)

            # 4. Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = student_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # --------------------------
        # Validation
        # --------------------------
        student_model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = student_model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )

    # ---------------------------------
    # Save Distilled Student Model
    # ---------------------------------
    torch.save(student_model.state_dict(), "student_resnet18_kd.pth")
    print("\nKnowledge Distillation Complete!")
    print("Student model saved as student_resnet18_kd.pth")

if __name__ == '__main__':
    main()
