# test_model.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
_, _, test_loader = load_datasets()


# ---------------------------------
# Load Model
# ---------------------------------
model = ResNet101_NonLocal(num_classes=2)

model.load_state_dict(torch.load("fruit_freshness_model.pth", map_location=device))

model = model.to(device)

model.eval()


# ---------------------------------
# Testing Loop
# ---------------------------------
all_preds = []
all_labels = []

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

precision = precision_score(all_labels, all_preds)

recall = recall_score(all_labels, all_preds)

f1 = f1_score(all_labels, all_preds)

cm = confusion_matrix(all_labels, all_preds)


print("\n----- Test Results -----")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)