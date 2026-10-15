
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
CLASSES = [
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam",
]

BASE = Path("/content/studio2_awal")
PROJECT = BASE / "proyek"
TRAIN_DIR = BASE / "data_split/train"
VAL_DIR = BASE / "data_split/val"
OUT = PROJECT / "experiments/augmentasi"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)

assert train_ds.classes == CLASSES, train_ds.classes
assert val_ds.classes == CLASSES, val_ds.classes

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Train:", len(train_ds), "| Validasi:", len(val_ds))

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

history = []
best_f1 = -1.0
best_epoch = 0
best_path = OUT / "model_augmentasi.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    train_loss = total_loss / len(train_ds)

    model.eval()
    val_loss_sum = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            val_loss_sum += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss = val_loss_sum / len(val_ds)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    scheduler.step()

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
    }
    history.append(row)

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_epoch = epoch
        torch.save(model.state_dict(), best_path)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f}"
    )

pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)

summary = {
    "experiment": "resnet18_stronger_augmentation",
    "epochs": EPOCHS,
    "selected_epoch": best_epoch,
    "selection_metric": "validation_macro_f1",
    "best_validation_macro_f1": best_f1,
    "class_order": CLASSES,
}
import json
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nEksperimen 2 selesai.")
print("Epoch terbaik:", best_epoch)
print("Macro-F1 terbaik:", round(best_f1, 4))
print("Checkpoint:", best_path)
