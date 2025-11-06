import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training auf {device.type}")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7, dropout_b=0.25, dropout_fc=0.5):
        super(SimpleCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # (64, 48, 48)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # (64, 48, 48)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # (64, 24, 24)
        self.dropout1 = nn.Dropout(dropout_b)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (128, 24, 24)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # (128, 24, 24)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # (128, 12, 12)
        self.dropout2 = nn.Dropout(dropout_b)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # (256, 12, 12)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # (256, 12, 12)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # (256, 6, 6)
        self.dropout3 = nn.Dropout(dropout_b)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_fc)

        self.fc2 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(dropout_fc)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = self.flatten(x)

        # Fully Connected
        x = F.leaky_relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)

        x = F.leaky_relu(self.bn8(self.fc2(x)))
        x = self.dropout5(x)

        x = self.fc3(x)

        return x


class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.min_val_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model=None):
        if self.min_val_loss is None:
            self.min_val_loss = val_loss
        elif val_loss >= self.min_val_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Counter increased: {self.counter - 1} ---> {self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if self.verbose:
                print(f"Val loss decreased: {self.min_val_loss:.4f} --> {val_loss:.4f}")
            self.min_val_loss = val_loss


def plot_and_log_confusion_matrix(all_targets, all_preds, classes, filename="confusion_matrix.png"):
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()
    os.remove(filename)


def plot_and_log_classification_report(all_targets, all_preds, classes, filename="classification_report.png"):
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    if 'support' in df_report.columns:
        df_report = df_report.drop(columns=['support'])
    if 'accuracy' in df_report.index:
        df_report = df_report.drop(index=['accuracy'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:-3, :], annot=True, cmap="YlGnBu", cbar=True, vmin=0, vmax=1)
    plt.title("Classification Report (Precision, Recall, F1)")
    plt.tight_layout()
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()
    os.remove(filename)


def plot_and_log_class_distribution(targets, classes, filename="train_class_dist.png"):
    unique, counts = np.unique(targets, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar([classes[i] for i in unique], counts, color='skyblue', edgecolor='black')
    plt.title("Train Class Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()
    os.remove(filename)


def plot_and_log_misclassified_examples(model, val_loader, classes, device, max_examples=8, filename="misclassified_examples.png"):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            incorrect = predicted != target
            if incorrect.any():
                mis_data = data[incorrect][:max_examples - len(misclassified)]
                mis_true = target[incorrect][:max_examples - len(misclassified)]
                mis_pred = predicted[incorrect][:max_examples - len(misclassified)]
                for i in range(len(mis_data)):
                    misclassified.append({
                        'image': mis_data[i].cpu(),
                        'true': mis_true[i].item(),
                        'pred': mis_pred[i].item()
                    })
                if len(misclassified) >= max_examples:
                    break

    if misclassified:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        for idx, item in enumerate(misclassified[:max_examples]):
            img = item['image'].squeeze()
            true_label = classes[item['true']]
            pred_label = classes[item['pred']]
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        os.remove(filename)


def objective(trial, mean, std, datapath):
    early_stopping = EarlyStopping(verbose=True)

    dropout_b = trial.suggest_float("dropout_b", 0.1, 0.3)
    dropout_fc = trial.suggest_float("dropout_fc", 0.2, 0.5)
    aug_p = trial.suggest_float("aug_p", 0.1, 0.5)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    use_label_smoothing = trial.suggest_categorical("use_label_smoothing", [False, True])
    label_smoothing = trial.suggest_float("label_smoothing", 0.01, 0.3) if use_label_smoothing else 0.0

    with mlflow.start_run(run_name=f"simple_cnn_{trial.number}", nested=True) as child_run:
        base_lr = 0.001
        epochs = 100

        hyperparams = {
            "dropout_b": dropout_b,
            "dropout_fc": dropout_fc,
            "aug_p": aug_p,
            "label_smoothing": label_smoothing,
            "weight_decay": wd,
            "learning_rate": base_lr,
            "epochs": epochs
        }

        model = SimpleCNN(dropout_b=dropout_b, dropout_fc=dropout_fc).to(device)
        mlflow.log_params(hyperparams)

        transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.RandomApply([T.RandAugment()], p=aug_p),
            T.Resize(48, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        train_dataset = datasets.ImageFolder(root=datapath / "train", transform=transform)
        val_dataset = datasets.ImageFolder(root=datapath / "test", transform=transform)

        # Logge Klassenverteilung
        train_targets = np.array(train_dataset.targets)
        plot_and_log_class_distribution(train_targets, train_dataset.classes)

        label_encoding = train_dataset.class_to_idx
        with open("label_encoding.json", "w") as f:
            json.dump(label_encoding, f, indent=4)
        mlflow.log_artifact("label_encoding.json")
        os.remove("label_encoding.json")

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        optimizer = torch.optim.Adam(lr=base_lr, weight_decay=wd, params=model.parameters())
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)

        for epoch in range(epochs):
            model.train()
            train_loader_bar = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch + 1}/{epochs}", leave=False)

            for data, target in train_loader_bar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    logits = model(data)
                    loss = criterion(logits, target)
                    val_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            epoch_accuracy = correct / total

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_accuracy, step=epoch)

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early Stopping triggered! (Uhrzeit: {datetime.now().strftime('%H:%M:%S')})")
                break

        # Finale Evaluation
        model.eval()
        total_correct = 0
        total_samples = 0
        final_preds = []
        final_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)
                final_preds.extend(predicted.cpu().numpy())
                final_targets.extend(target.cpu().numpy())

        accuracy = total_correct / total_samples

        plot_and_log_confusion_matrix(final_targets, final_preds, train_dataset.classes)
        plot_and_log_classification_report(final_targets, final_preds, train_dataset.classes)
        plot_and_log_misclassified_examples(model, val_loader, train_dataset.classes, device)

        model_filename = f"model_trial_{trial.number}.pth"
        torch.save(model.state_dict(), model_filename)
        mlflow.log_artifact(model_filename)

        return accuracy, model_filename


def optimize_model(mean, std, n_trials, datapath):
    with mlflow.start_run(run_name="SimpleCNN", nested=False) as parent_run:

        
        pbar = tqdm(total=n_trials, desc="Optuna Trials", unit="trial")

        def update_progress(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Best Acc": f"{study.best_value:.4f}" if study.best_value else "N/A"})

        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(),
            direction="maximize",
            study_name="SimpleCNN_study"
        )

        def objective_wrapper(trial):
            acc, _ = objective(trial, mean, std, datapath)
            return acc

        study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[update_progress])

        pbar.close()  # Wichtig: schließe die Bar am Ende

        print(f"\n✅ Best Trial: {study.best_trial.number}")
        print(f"🎯 Best Params: {study.best_params}")
        print(f"🏆 Best accuracy: {study.best_value:.4f}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_acc", study.best_value, run_id=parent_run.info.run_id)

        # Optuna Plots
        try:
            import optuna.visualization as vis

            plots = {
                "optimization_history.png": vis.plot_optimization_history(study),
                "param_importances.png": vis.plot_param_importances(study),
                "parallel_coordinate.png": vis.plot_parallel_coordinate(study),
                "slice_plot.png": vis.plot_slice(study),
                "contour_plot.png": vis.plot_contour(study)
            }

            for filename, fig in plots.items():
                fig.write_image(filename)
                mlflow.log_artifact(filename)
                os.remove(filename)

        except ImportError:
            print("⚠️ Plotly nicht installiert. Installiere mit: pip install plotly")

        # Lade bestes Modell
        best_trial_number = study.best_trial.number
        best_model_filename = f"model_trial_{best_trial_number}.pth"

        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name("SIMPLECNN")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = 'simple_cnn_{best_trial_number}'",
            max_results=1
        )

        if runs:
            best_run = runs[0]
            local_model_path = mlflow.artifacts.download_artifacts(
                run_id=best_run.info.run_id,
                artifact_path=best_model_filename
            )

            best_model = SimpleCNN(
                dropout_b=study.best_params["dropout_b"],
                dropout_fc=study.best_params["dropout_fc"]
            )
            best_model.load_state_dict(torch.load(local_model_path, map_location="cpu"))

            final_model_path = Path("./SimpleCNNModel/best_final_model.pth")
            final_model_path.parent.mkdir(exist_ok=True)
            torch.save(best_model.state_dict(), final_model_path)
            mlflow.log_artifact(final_model_path)

            print(f"\n💾 Bestes Modell gespeichert als '{final_model_path}' und in MLflow geloggt.")
        else:
            print("❌ Fehler: Konnte besten Run nicht finden.")


if __name__ == "__main__":
    mlflow.set_experiment("SIMPLECNN")

    datapath = Path.cwd() / "data"
    mean = 0.5077
    std = 0.2551
    n_trials = 30

    optimize_model(mean, std, n_trials, datapath)