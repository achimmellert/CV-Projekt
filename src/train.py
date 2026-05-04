import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from src.models.cnn import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0005, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.min_val_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.min_val_loss is None:
            self.min_val_loss = val_loss
        elif val_loss >= self.min_val_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.min_val_loss = val_loss
            self.counter = 0


def objective(trial, mean, std, datapath):

    dropout_b = trial.suggest_float("dropout_b", 0.05, 0.2)
    dropout_fc = trial.suggest_float("dropout_fc", 0.3, 0.6)
    aug_p = trial.suggest_float("aug_p", 0.2, 0.5)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    base_lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    use_label_smoothing = trial.suggest_categorical("use_label_smoothing", [True, False])
    ls_val = trial.suggest_float("label_smoothing", 0.05, 0.15) if use_label_smoothing else 0.0

    epochs = 50
    batch_size = 128

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)

        transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.RandomApply([T.RandAugment(num_ops=2, magnitude=9)], p=aug_p),
            T.Resize((48, 48)),
            T.ToTensor(),
            T.Normalize(mean=[mean], std=[std])
        ])

        train_ds = datasets.ImageFolder(root=datapath / "train", transform=transform)
        val_ds = datasets.ImageFolder(root=datapath / "test", transform=transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        model = CNN(dropout_b=dropout_b, dropout_fc=dropout_fc).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss(label_smoothing=ls_val)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=base_lr, steps_per_epoch=len(train_loader), epochs=epochs
        )

        # Mixed Precision Scaler
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        early_stopping = EarlyStopping(patience=8)

        for epoch in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Forward Pass mit Autocast
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    logits = model(data)
                    loss = criterion(logits, target)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                        logits = model(data)
                        val_loss += criterion(logits, target).item()
                        _, predicted = torch.max(logits, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            acc = correct / total

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", acc, step=epoch)

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                break

        model_path = f"model_trial_{trial.number}.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        os.remove(model_path)

        return acc


def optimize_model(datapath, n_trials=30):
    mean, std = 0.5077, 0.2551

    with mlflow.start_run(run_name="EmotionResNet"):
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
        study.optimize(lambda t: objective(t, mean, std, datapath), n_trials=n_trials)

        print(f"Beste Genauigkeit: {study.best_value:.4f}")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_acc", study.best_value)


if __name__ == "__main__":
    mlflow.set_experiment("EMOTION_RESNET")
    optimize_model(Path("data"))
