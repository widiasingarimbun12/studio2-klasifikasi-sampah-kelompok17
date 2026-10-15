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
TRAIN_DIR = BASE / "data_split/train"
VAL_DIR = BASE / "data_split/val"
OUT = BASE / "proyek"
SUBMISSION = OUT / "submission"
SUBMISSION.mkdir(parents=True, exist_ok=True)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Perangkat:", device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    print("Kelas:", train_ds.classes)
    assert train_ds.classes == CLASSES
    assert val_ds.classes == CLASSES

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_f1 = -1
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_ds)
        model.eval()
        y_true, y_pred = [], []
        val_loss_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(outputs.argmax(1).cpu().tolist())

        val_loss = val_loss_total / len(val_ds)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": acc,
            "val_macro_f1": macro_f1,
        })

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"accuracy={acc:.4f} | macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), SUBMISSION / "model.pth")
            print("Model terbaik disimpan.")

    pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)
    print("Pelatihan selesai. Macro-F1 terbaik:", round(best_f1, 4))
    print("Model:", SUBMISSION / "model.pth")

if __name__ == "__main__":
    main()
