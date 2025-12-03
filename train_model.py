# train_model.py
#
# Train a small CNN on 16x16 RGB patches saved by combined_outline.py.
# Dataset expected at: dataset/patch_dataset.npz with:
#   X: (N, 256, 3) uint8  -> 16x16 RGB patches flattened
#   y: (N,) int64          -> 0=PLASTIC, 1=METAL
#
# The best model (by validation accuracy) is saved to models/material_cnn.pth
# and can be reloaded from outline.py.

import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------------- Config ----------------------

DATASET_PATH = "dataset/patch_dataset_2.npz"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "material_cnn_2.pth")

BATCH_SIZE = 64
EPOCHS = 15          # small epoch count as requested
LR = 1e-3
WEIGHT_DECAY = 1e-4  # small L2 regularization
VAL_SPLIT = 0.2
PATIENCE = 4         # early stopping patience

# label mapping (must match combined_outline.py)
LABEL_MAP = {
    0: "PLASTIC",
    1: "METAL",
}

# ---------------------- Model ----------------------


class MaterialPatchNet(nn.Module):
    """
    Small CNN tuned for 16x16 RGB patches.
    Input:  (N, 3, 16, 16)
    Output: logits for 2 classes (plastic, metal)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.3)

        # After 3 conv + 2x2 pools, spatial size: 16 -> 8 -> 4 -> 2
        # so features are (128, 2, 2) = 512
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (N, 3, 16, 16)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 16 -> 8

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 8 -> 4

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 4 -> 2

        x = x.view(x.size(0), -1)  # (N, 512)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------- Dataset ----------------------


class PatchDataset(Dataset):
    """
    Wraps X (N, 256, 3) uint8 patches and y (N,) labels.
    Provides tensors shaped (3, 16, 16) with [0,1] scaling and slight augmentation.
    """

    def __init__(self, X, y, indices, augment: bool = False):
        self.X = X
        self.y = y
        self.indices = np.array(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        patch = self.X[real_idx]  # (256, 3) uint8
        label = int(self.y[real_idx])

        # reshape to (3, 16, 16) and scale to [0,1]
        patch = patch.reshape(16, 16, 3).astype("float32") / 255.0
        patch = np.transpose(patch, (2, 0, 1))  # (3, 16, 16)

        patch = torch.from_numpy(patch)

        if self.augment:
            patch = self._augment(patch)

        return patch, label

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """
        Slight randomness:
        - random horizontal/vertical flips
        - slight brightness + contrast jitter
        - tiny gaussian noise
        All operations keep img in [0,1] range.
        """
        # Flips
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])  # horizontal flip
        if random.random() < 0.1:
            img = torch.flip(img, dims=[1])  # vertical flip (less frequent)

        # Brightness / contrast jitter
        if random.random() < 0.3:
            # brightness
            factor = 1.0 + 0.1 * (2 * random.random() - 1)  # in [0.9, 1.1]
            img = img * factor
            # contrast (center around mean)
            if random.random() < 0.5:
                mean = img.mean()
                c_factor = 1.0 + 0.2 * (2 * random.random() - 1)  # [0.8, 1.2]
                img = (img - mean) * c_factor + mean

        # Gaussian noise (tiny)
        if random.random() < 0.3:
            noise = torch.randn_like(img) * 0.02
            img = img + noise

        img = torch.clamp(img, 0.0, 1.0)
        return img


# ---------------------- Helpers ----------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_balanced_sampler(labels_subset):
    """
    Create a WeightedRandomSampler to compensate for class imbalance.
    """
    labels_subset = np.array(labels_subset)
    class_counts = np.bincount(labels_subset)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels_subset]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels_subset),
        replacement=True,
    )
    return sampler


# ---------------------- Training Loop ----------------------


def train():
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. "
                                f"Run combined_outline.py --test all first.")

    data = np.load(DATASET_PATH)
    X = data["X"]  # (N, 256, 3)
    y = data["y"]  # (N,)

    N = X.shape[0]
    if N == 0:
        raise ValueError("Empty dataset: no patches found in dataset/patch_dataset_2.npz")

    print(f"Loaded dataset with {N} patches.")
    print("Class balance:", {k: int(np.sum(y == k)) for k in np.unique(y)})

    # Train / val split
    indices = np.arange(N)
    np.random.shuffle(indices)
    split = int(N * (1 - VAL_SPLIT))
    train_indices = indices[:split]
    val_indices = indices[split:]

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # Datasets
    train_ds = PatchDataset(X, y, train_indices, augment=True)
    val_ds = PatchDataset(X, y, val_indices, augment=False)

    # Sampler for class balance on training set
    train_labels_subset = y[train_indices]
    train_sampler = create_balanced_sampler(train_labels_subset)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model, optimizer, loss
    model = MaterialPatchNet(num_classes=len(LABEL_MAP)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler (cosine decay for better convergence)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss /= max(1, total)
        train_acc = correct / max(1, total)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                _, preds = torch.max(logits, dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        # ---- Checkpointing / early stopping ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_no_improve = 0
            print(f"  * New best val_acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Save best model
    if best_state is None:
        best_state = model.state_dict()

    checkpoint = {
        "state_dict": best_state,
        "label_map": LABEL_MAP,
        "input_shape": [3, 16, 16],
        "meta": {
            "epochs_trained": epoch + 1,
            "best_val_acc": best_val_acc,
        },
    }

    torch.save(checkpoint, MODEL_PATH)
    print(f"\nSaved best model to {MODEL_PATH}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    # Slight randomness but reproducible-ish base
    set_seed(42)
    train()
