#!/usr/bin/env python3
"""
JetRacer imitation-learning trainer  (run on server, not on Jetson)

Dataset layout expected:
    <data_dir>/
        section_1/
            150_150_000000_20260101120000000.jpg   ← cx=150 (직진)
            080_080_000001_20260101120000100.jpg   ← cx=80  (좌회전)
            ...
        section_2/
            ...

이미지는 data_collection_node가 저장한 그대로 사용됨 — 크롭 여부/비율은
params/params.yaml의 crop_ratio에 달려있고, 여기서는 그 결과물을
추가 크롭 없이 224×224로 리사이즈만 수행.

Label encoded in filename:  {cx:03d}_{cx:03d}_{index:06d}_{timestamp}.jpg
    cx ∈ [0, 300], center=150
    normalized label = (150 - cx) / 150   ∈ [-1, 1]
    angular_z = label * max_steer_angle

Usage:
    python train.py --data_dir ~/jetracer_dataset --section 1 --epochs 50
    python train.py --data_dir ~/jetracer_dataset --section all --epochs 50
"""

import argparse
import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class SectionDataset(Dataset):
    """경로 리스트를 받아 이미지와 조향각 라벨을 반환."""

    def __init__(self, paths: list, transform=None, augment_flip: bool = False):
        self.paths = paths
        self.transform = transform
        self.augment_flip = augment_flip

    def __len__(self):
        # flip augmentation: 같은 이미지를 원본 + 좌우반전 으로 2배 사용
        return len(self.paths) * (2 if self.augment_flip else 1)

    def __getitem__(self, idx):
        flip = self.augment_flip and idx >= len(self.paths)
        path = self.paths[idx % len(self.paths)]

        cx = int(os.path.basename(path).split('_')[0])
        label = (150.0 - cx) / 150.0  # ∈ [-1, 1]
        if flip:
            label = -label

        # data_collector가 이미 크롭해서 저장 → 추가 크롭 불필요
        img = Image.open(path).convert('RGB')
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def _make_transform(augment: bool):
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if augment:
        base = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
        ] + base
    return transforms.Compose(base)


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #

def train_section(data_dir: str, section_id: int, epochs: int,
                  batch_size: int, lr: float, output_dir: str,
                  device: torch.device):
    section_dir = os.path.join(data_dir, f'section_{section_id}')
    all_paths = sorted(glob.glob(os.path.join(section_dir, '*.jpg')))
    if not all_paths:
        print(f'[section {section_id}] No images in {section_dir} — skipping.')
        return

    # 경로 레벨에서 먼저 셔플 후 train/val 분리
    # → flip 쌍(원본+반전)이 train/val에 동시에 들어가는 누수 방지
    random.seed(42)
    random.shuffle(all_paths)
    n_val = max(1, int(len(all_paths) * 0.1))
    val_paths   = all_paths[:n_val]
    train_paths = all_paths[n_val:]
    n_train = len(train_paths)   # flip 적용 전 실제 이미지 수

    # 별도 Dataset 인스턴스 → transform이 공유되지 않음
    train_ds = SectionDataset(train_paths,
                              transform=_make_transform(augment=True),
                              augment_flip=True)   # 좌우반전으로 2배
    val_ds   = SectionDataset(val_paths,
                              transform=_make_transform(augment=False),
                              augment_flip=False)  # val은 augmentation 없음

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, f'section_{section_id}.pth')
    best_val = float('inf')

    print(f'\n[section {section_id}]  train={n_train}(×2flip={n_train*2})  val={n_val}  '
          f'device={device}')

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs).squeeze(1)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)  # flip 포함 실제 샘플 수

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs).squeeze(1)
                val_loss += criterion(pred, labels).item() * imgs.size(0)
        val_loss /= len(val_ds)
        scheduler.step()

        print(f'  epoch {epoch:3d}/{epochs}  '
              f'train={train_loss:.6f}  val={val_loss:.6f}', end='')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f'  ✓ saved', end='')
        print()

    # Export ONNX from best checkpoint
    _export_onnx(model, best_path,
                 os.path.join(output_dir, f'section_{section_id}.onnx'),
                 device)


def _export_onnx(model: nn.Module, pth_path: str,
                 onnx_path: str, device: torch.device):
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11,
        do_constant_folding=True,
    )
    print(f'  ONNX exported → {onnx_path}')


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='JetRacer section model trainer')
    parser.add_argument('--data_dir', required=True,
                        help='Root of dataset (contains section_N/ folders)')
    parser.add_argument('--section', default='all',
                        help='Section id 1-5, or "all"')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='./models',
                        help='Where to save .pth and .onnx files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    sections = list(range(1, 6)) if args.section == 'all' else [int(args.section)]

    for sec in sections:
        train_section(args.data_dir, sec, args.epochs,
                      args.batch_size, args.lr, args.output_dir, device)


if __name__ == '__main__':
    main()
