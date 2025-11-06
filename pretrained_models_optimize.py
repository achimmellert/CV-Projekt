from fastai.vision.all import *
import optuna
import mlflow
from fasttransform import Transform
from optuna.pruners import HyperbandPruner
from datetime import timedelta
import optuna.visualization as vis
import os
import torch
import shutil
import torchvision.transforms as T
from torchvision import datasets
from tqdm import tqdm
import logging


# Sicherstellen, dass Ordner existieren
os.makedirs("./logs", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="./logs/loggings.log"
)
logger = logging.getLogger(name=__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training auf {device.type}")


def load_data():
    logger.info("Loading data....")

    import kagglehub
    path = kagglehub.dataset_download("msambare/fer2013")

    dest = Path.cwd() / "data"
    shutil.copytree(src=path, dst=dest, dirs_exist_ok=True)


# Calculating mean and std for FER2013
def calculate_mean_std():
    logger.info("Calculating mean and std for Normalization....")


    total_sum = 0.
    total_sq = 0.
    num_pixels = 0

    transform = T.Compose([
        T.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=Path.cwd() / "data" / "train", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    for images, _ in tqdm(loader, desc="Berechne Mean und Std", total=len(loader)):
        total_sum += images.sum()  # Pixelwerte addiert
        total_sq += (images ** 2).sum()  # Quadrierte Pixelwerte addiert
        num_pixels += images.numel()  # Anzahl Pixels (64 × 1 × 48 × 48)

    mean = total_sum / num_pixels  # E[X]
    variance = total_sq / num_pixels - mean ** 2  # E[X^2] - (E[X])^2
    std = variance ** 0.5  # (variance)^2

    print(mean)
    print(std)
    return mean.item(), std.item()


# Method Overloading via Type Dispatching
class To3Channels(Transform):
    def encodes(self, x: PILImageBW):
        return x.convert("RGB")
    def encodes(self, x: TensorImageBW):
        return x.repeat(3, 1, 1)


def create_dls(datapath: Path, bs, aug_p) -> DataLoaders:
    logger.info("Creating dls....")

    dblock = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(train_name="train", valid_name="test"),
        item_tfms=[
            Resize((224, 224)), # Modelle wurden mit ImageNet trainiert (Dimension 224x224)
            To3Channels
        ],
        batch_tfms=[
            Brightness(p=aug_p),
            Contrast(p=aug_p),
            Rotate(p=aug_p),
            Normalize.from_stats(*imagenet_stats)
        ]
        )
    dls = dblock.dataloaders(datapath, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)

    return dls

# Closure für Optuna-Callback
# Gibt die richtige Callback-Funktion an Optuna zurück
"""def create_plot_callback(model_name):
    def callback(study, trial):
        if trial.number % 5 == 0:
            plot_optuna(study, model_name)
    return callback"""


# Optuna Plots sind auf Study-Ebene, nicht auf Trial-Ebene
def plot_optuna(study, model_name):
    """
    Loggt die aktuellen Optuna-Visualisierungen in den aktiven MLflow-Run.
    Kann nach jedem Trial aufgerufen werden.
    """
    plot_configs = [
        ("optimization_history", vis.plot_optimization_history),
        ("param_importances", vis.plot_param_importances),
        ("slice_plot", vis.plot_slice),
        ("contour_plot", vis.plot_contour)
    ]

    for name, plot_func in plot_configs:
        # Generiere Plot
        fig = plot_func(study)

        # Dateiname
        filename = f"{model_name}_{name}.png"

        # Speichern (Plotly → write_image)
        fig.write_image(filename, width=1200, height=800)

        # Loggen in MLflow (ARTIFACT_PATH: z.B. "optuna_live_plots")
        mlflow.log_artifact(filename, artifact_path="optuna_live_plots")

        os.remove(filename)


def objective(trial, model_name):
    logger.info(f"##### Trial {trial.number}")

    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    aug_p = trial.suggest_float("aug_p", 0.1, 0.5, step=0.1)
    dropout_p = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    # lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    with mlflow.start_run(run_name=f"{model_name}_{trial.number}", nested=True) as child_run:
        hyperparams={
            "weight_decay": wd,
            "batch_size": 256,
            "augmentation_p": aug_p,
            "dropout_p": dropout_p,
            "learning_rate": 0.001
        }
        mlflow.log_params(hyperparams)

        dls = create_dls(datapath=Path.cwd() / "data", bs=256, aug_p=aug_p)

        model = vision_learner(
            dls=dls,
            arch=model_name,
            metrics=[accuracy, F1Score(average="weighted")],
            cbs=[
                EarlyStoppingCallback(monitor="valid_loss", patience=5, min_delta=0.001),
                SaveModelCallback(monitor="accuracy", fname=f"best_{model_name}_trial_{trial.number}"), # Nur das beste Modell wird gespeichert
                MixedPrecision()
            ],
            ps=dropout_p,
            model_dir=Path("models") / model_name,
        )


        model.fine_tune(base_lr=0.001, wd=wd, epochs=15)
        logger.info("Trial-Training beendet.")

        results = model.validate()
        metrics = {}
        metrics["valid_loss"] = results[0]
        metrics["acc"] = results[1]
        metrics["f1score"] = results[2]
        mlflow.log_metrics(metrics)

        model_path = Path("models") / model_name / f"best_{model_name}_trial_{trial.number}.pth"
        if model_path.exists():
            mlflow.log_artifact(model_path, artifact_path="models")
            logger.info(f"Bestes Modell gespeichert: {model_path}")
        else:
            logger.warning(f"Modell {model_path} wurde nicht gefunden!")

        interp = ClassificationInterpretation.from_learner(model)
        interp.plot_confusion_matrix()
        fig = plt.gcf()
        confusion_matrix_path = "confusion_matrix.png"
        fig.savefig(confusion_matrix_path, dpi=300)
        mlflow.log_artifact(confusion_matrix_path, "Interpretation")
        plt.close(fig)

        return metrics["acc"]


def optimize_models(model_names: list[str], n_trials):
    for model_name in model_names:
        os.makedirs(f"./models/{str(model_name)}", exist_ok=True)

        with mlflow.start_run(nested=False, run_name=f"Study_{model_name}") as parent_run:
            study = optuna.create_study(
                pruner=HyperbandPruner(),
                direction="maximize",
                study_name=f"{model_name}_study"
            )

            start_time = time.time()
            study.optimize(
                lambda trial: objective(trial, model_name),
                n_trials=n_trials,
            )

            plot_optuna(study, model_name)


            duration = time.time() - start_time
            avg_trial_time = duration / n_trials
            remaining_models = len(model_names) - (model_names.index(model_name) + 1)
            estimated_remaining = remaining_models * n_trials * avg_trial_time
            logger.info(f"Beendet nach: {timedelta(seconds=duration)}")
            logger.info(f"Verbleibende Zeit geschätzt: {timedelta(seconds=estimated_remaining)}")

            logger.info(f"Best Trial: {study.best_trial}")
            logger.info(f"Best Params: {study.best_params}")
            logger.info(f"Best accuracy: {study.best_value}")

            mlflow.log_params(study.best_params)

            mlflow.log_metric("best_acc", study.best_value, run_id=parent_run.info.run_id)

            best_model_fname = f"best_{model_name}_trial_{study.best_trial.number}.pth"
            best_model_path = Path("models") / model_name / best_model_fname

            # Kopiere / logge es als finales bestes Modell
            final_model_path = Path("models") / f"best_{model_name}_overall.pth"
            shutil.copy(best_model_path, final_model_path)
            mlflow.log_artifact(final_model_path, artifact_path="final_model")
            logger.info(f"Bestes Modell der gesamten Study: {final_model_path}")


if __name__ == "__main__":
    mlflow.set_experiment("Models_testen_fer2013")
    load_data()
    models_to_test = ["efficientnet_b0", "convnext_tiny"]
    n_trials = 5
    optimize_models(models_to_test, n_trials)
