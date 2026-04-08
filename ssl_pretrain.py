import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# ------------------------------------
# Device
# ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------
# SimCLR Dataset Wrapper
# ------------------------------------
class SimCLRDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)

        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2


# ------------------------------------
# Strong Augmentation
# ------------------------------------
ssl_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------
# Load Dataset
# ------------------------------------
dataset = SimCLRDataset(
    root="dataset/train",
    transform=ssl_transform
)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # keep 0 for Windows stability
    pin_memory=True
)


# ------------------------------------
# Projection Head
# ------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------
# SimCLR Model
# ------------------------------------
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return z


# ------------------------------------
# NT-Xent Loss
# ------------------------------------
def contrastive_loss(z1, z2, temperature=0.5):

    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim_matrix = torch.matmul(z, z.T)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.cat([
        torch.sum(z1 * z2, dim=-1),
        torch.sum(z2 * z1, dim=-1)
    ], dim=0)

    positives = torch.exp(positives / temperature)

    negatives = torch.sum(torch.exp(sim_matrix / temperature), dim=-1)

    loss = -torch.log(positives / negatives)

    return loss.mean()


# ------------------------------------
# Training
# ------------------------------------
def main():

    print("Starting training...")

    model = SimCLR().to(device)
    print("Model loaded")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):

        print(f"Starting Epoch {epoch+1}")

        model.train()
        total_loss = 0

        for batch_idx, (x1, x2) in enumerate(train_loader):

            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")


#_-__-_-____---_-___--_--_-----_---_____------
if __name__ == "__main__":
    main()