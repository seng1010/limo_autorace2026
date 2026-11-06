#!/usr/bin/env python3
"""
JetRacer go/stop 멀티태스크(조향+정지) 학습기  (GPU 서버에서 실행, 단독 실행 가능)

Dataset layout expected:
    <data_dir>/
        go_stop/
            stop/   ← 정지 지점 근처 사진 (파일명에 조향 cx 라벨 포함)
            go/     ← 나머지 주행 사진 (파일명에 조향 cx 라벨 포함)

go/, stop/ 두 폴더 모두 data_collection_node가 저장하는 것과 같은 파일명 관례
({cx:03d}_{cx:03d}_{index:06d}_{timestamp}.jpg)를 따른다고 가정한다.
    - 조향 라벨: 파일명의 cx (모든 이미지에서 뽑음, train.py와 동일한 공식)
    - 정지 라벨: 어느 폴더에 있는지 (stop=1.0, go=0.0)

모델은 출력 2개짜리 ResNet18이다:
    output[0] = steering (회귀, train.py 라벨 관례와 동일: +1=완전좌회전, -1=완전우회전)
    output[1] = stop raw logit (분류, BCEWithLogitsLoss로 학습 — sigmoid는 추론 노드에서 씌움)
두 손실(MSE + BCE)을 더해서 하나의 loss로 같이 학습한다 (stop_loss_weight로
정지쪽 비중 조절 가능).

매 epoch마다 steer_mse(조향 오차)와 stop_recall/go_falsestop(정지 판단 지표 —
1.0/0.0에 가까울수록 각각 좋음, "엉뚱한 데서 멈춤" 리스크 지표)을 같이 출력한다.

Usage:
    python train_go_stop.py --data_dir ~/jetracer_dataset --epochs 50
"""

import argparse
import glob
import os
import random

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ------------------------------------------------------------------ #
# Model / transform / export
# ------------------------------------------------------------------ #


def build_model() -> nn.Module:
    """출력 2개: [0]=steering(회귀), [1]=stop raw logit(분류)."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


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
# Dataset
# ------------------------------------------------------------------ #


class GoStopDataset(Dataset):
    """(path, stop_label) 리스트를 받아 이미지, 조향 라벨(파일명 cx), 정지 라벨을 반환."""

    def __init__(self, samples: list, transform=None, augment_flip: bool = False):
        self.samples = samples  # [(path, stop_label), ...]
        self.transform = transform
        self.augment_flip = augment_flip

    def __len__(self):
        # flip augmentation: 같은 이미지를 원본 + 좌우반전 으로 2배 사용
        return len(self.samples) * (2 if self.augment_flip else 1)

    def __getitem__(self, idx):
        flip = self.augment_flip and idx >= len(self.samples)
        path, stop_label = self.samples[idx % len(self.samples)]

        cx = int(os.path.basename(path).split('_')[0])
        steer_label = (150.0 - cx) / 150.0  # ∈ [-1, 1]
        if flip:
            steer_label = -steer_label
            # 정지 여부(stop_label)는 좌우반전과 무관하므로 그대로 둔다.

        img = Image.open(path).convert('RGB')
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor([steer_label, stop_label], dtype=torch.float32)
        return img, label


def _split(paths: list, val_ratio: float = 0.1):
    """클래스별로 따로 나눠서 val에서도 stop/go 비율이 유지되게 한다."""
    random.seed(42)
    paths = list(paths)
    random.shuffle(paths)
    n_val = max(1, int(len(paths) * val_ratio))
    return paths[n_val:], paths[:n_val]  # train, val


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #


def train_go_stop(data_dir: str, epochs: int, batch_size: int, lr: float,
                  output_dir: str, stop_loss_weight: float, device: torch.device):
    go_stop_dir = os.path.join(data_dir, 'go_stop')
    stop_paths = sorted(glob.glob(os.path.join(go_stop_dir, 'stop', '*.jpg')))
    go_paths = sorted(glob.glob(os.path.join(go_stop_dir, 'go', '*.jpg')))

    if not stop_paths or not go_paths:
        print(f'[go_stop] stop={len(stop_paths)}장  go={len(go_paths)}장 — '
              f'{go_stop_dir}/stop 와 {go_stop_dir}/go 둘 다 사진이 있어야 합니다.')
        return

    stop_train, stop_val = _split(stop_paths)
    go_train, go_val = _split(go_paths)

    train_samples = [(p, 1.0) for p in stop_train] + [(p, 0.0) for p in go_train]
    val_samples = [(p, 1.0) for p in stop_val] + [(p, 0.0) for p in go_val]

    train_ds = GoStopDataset(train_samples, transform=_make_transform(augment=True),
                             augment_flip=True)   # 좌우반전으로 2배
    val_ds = GoStopDataset(val_samples, transform=_make_transform(augment=False),
                           augment_flip=False)    # val은 augmentation 없음

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, 'go_stop.pth')
    best_val = float('inf')

    print(f'\n[go_stop]  stop: train={len(stop_train)} val={len(stop_val)}  '
          f'go: train={len(go_train)} val={len(go_val)}  device={device}')

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            steer_target, stop_target = labels[:, 0], labels[:, 1]

            optimizer.zero_grad()
            out = model(imgs)
            steer_pred, stop_logit = out[:, 0], out[:, 1]

            loss = mse(steer_pred, steer_target) + stop_loss_weight * bce(stop_logit, stop_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # --- val ---
        model.eval()
        val_loss = 0.0
        steer_sq_err = 0.0
        tp = fn = fp = tn = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                steer_target, stop_target = labels[:, 0], labels[:, 1]

                out = model(imgs)
                steer_pred, stop_logit = out[:, 0], out[:, 1]

                loss = mse(steer_pred, steer_target) + stop_loss_weight * bce(stop_logit, stop_target)
                val_loss += loss.item() * imgs.size(0)
                steer_sq_err += ((steer_pred - steer_target) ** 2).sum().item()

                preds = (torch.sigmoid(stop_logit) >= 0.5).float()
                tp += ((preds == 1) & (stop_target == 1)).sum().item()
                fn += ((preds == 0) & (stop_target == 1)).sum().item()
                fp += ((preds == 1) & (stop_target == 0)).sum().item()
                tn += ((preds == 0) & (stop_target == 0)).sum().item()
        val_loss /= len(val_ds)
        steer_mse = steer_sq_err / len(val_ds)
        scheduler.step()

        stop_recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        go_falsestop = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

        print(f'  epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  '
              f'steer_mse={steer_mse:.6f}  stop_recall={stop_recall:.3f}  '
              f'go_falsestop={go_falsestop:.3f}', end='')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f'  ✓ saved', end='')
        print()

    _export_onnx(model, best_path, os.path.join(output_dir, 'go_stop.onnx'), device)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(
        description='JetRacer go/stop multitask (steering + stop) trainer')
    parser.add_argument('--data_dir', required=True,
                        help='Root of dataset (contains go_stop/stop, go_stop/go)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='./models',
                        help='Where to save go_stop.pth / go_stop.onnx')
    parser.add_argument('--stop_loss_weight', type=float, default=1.0,
                        help='정지 분류 loss 비중 (조향 MSE 대비)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_go_stop(args.data_dir, args.epochs, args.batch_size, args.lr,
                 args.output_dir, args.stop_loss_weight, device)


if __name__ == '__main__':
    main()
