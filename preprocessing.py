# preprocessing.py

import os
import shutil
import random
from sklearn.model_selection import train_test_split

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------
# STEP 1: Dataset Split Function
# --------------------------------
def prepare_dataset(base_dir= r"C:\Users\hp\Desktop\GithubStuff\Fruit-Freshness-Detection\Dataset", output_dir="dataset"):
    print("Current working directory:", os.getcwd())

    if os.path.exists(os.path.join(output_dir, "train")):
        print("Dataset already prepared.")
        return

    print("Preparing dataset...")

    fresh_paths = []
    rotten_paths = []

    # Collect all image paths
    for category in os.listdir(base_dir):

        category_path = os.path.join(base_dir, category)

        if category == "Fresh":
            for sub in os.listdir(category_path):
                sub_path = os.path.join(category_path, sub)

                for img in os.listdir(sub_path):
                    fresh_paths.append(os.path.join(sub_path, img))

        elif category == "Rotten":
            for sub in os.listdir(category_path):
                sub_path = os.path.join(category_path, sub)

                for img in os.listdir(sub_path):
                    rotten_paths.append(os.path.join(sub_path, img))

    # Labels
    fresh_labels = [0] * len(fresh_paths)
    rotten_labels = [1] * len(rotten_paths)

    all_paths = fresh_paths + rotten_paths
    all_labels = fresh_labels + rotten_labels

    # Train + Temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )

    # Val + Test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Create folders
    for split in ["train", "val", "test"]:
        for cls in ["fresh", "rotten"]:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Copy files
    def copy_files(paths, labels, split):

        for path, label in zip(paths, labels):

            cls = "fresh" if label == 0 else "rotten"

            filename = os.path.basename(path)

            dst = os.path.join(output_dir, split, cls, filename)

            shutil.copy(path, dst)

    copy_files(X_train, y_train, "train")
    copy_files(X_val, y_val, "val")
    copy_files(X_test, y_test, "test")

    print("Dataset prepared successfully!")


# --------------------------------
# STEP 2: Transforms
# --------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------
# STEP 3: Load Dataset
# --------------------------------
def load_datasets(batch_size=32):

    # prepare_dataset()  # ← automatically prepares dataset #COMMENTED OUT CAUSE I DONT WANT TO RISK MAKING DUPLICATE DATASETS BY MISTAKE

    train_dataset = datasets.ImageFolder(
        root="dataset/train",
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root="dataset/val",
        transform=val_transforms
    )

    test_dataset = datasets.ImageFolder(
        root="dataset/test",
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
