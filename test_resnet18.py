import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import load_datasets

def get_student_model(num_classes=2):
    model = models.resnet18(weights=None)
    # Change the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def main():
    # ---------------------------------
    # Device Setup
    # ---------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    _, _, test_loader = load_datasets()

    # ---------------------------------
    # Load Model
    # ---------------------------------
    print("Initializing Student Model (ResNet18)...")
    model = get_student_model(num_classes=2)

    try:
        model.load_state_dict(torch.load("student_resnet18_kd.pth", map_location=device))
        print("Successfully loaded distilled student weights.")
    except Exception as e:
        print(f"Error loading student weights: {e}")
        print("Make sure you run train_kd_resnet18.py first to generate the weights.")
        return

    model = model.to(device)
    model.eval()

    # ---------------------------------
    # Testing Loop
    # ---------------------------------
    all_preds = []
    all_labels = []

    print("\nRunning inference on the test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ---------------------------------
    # Compute Metrics
    # ---------------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n----- Test Results (ResNet18 Edge Model) -----")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

if __name__ == '__main__':
    main()
