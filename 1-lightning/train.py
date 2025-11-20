import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class LitClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, hidden_dim=128):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss


def get_dataloaders(batch_size=64, val_split=0.1667):
    transform = transforms.ToTensor()

    # Load full training set
    full_train = datasets.MNIST(
        "./data", download=True, train=True, transform=transform
    )

    # Split into train and val
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 512)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    train_loader, val_loader = get_dataloaders(batch_size)

    model = LitClassifier(lr=lr, hidden_dim=hidden_dim)

    mlflow_logger = MLFlowLogger(
        experiment_name="mnist_optuna_lightning1",
        run_name=f"trial_{trial.number}",
        tracking_uri="sqlite:///mlflow_artifacts/mlruns.db",
        save_dir="./mlflow_artifacts",
    )

    mlflow_logger.log_hyperparams(
        {"lr": lr, "hidden_dim": hidden_dim, "batch_size": batch_size},
    )

    trainer = pl.Trainer(
        logger=mlflow_logger,
        max_epochs=10,
        accelerator="cpu",
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        enable_progress_bar=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    val_loss = trainer.callback_metrics["val_loss"].item()

    mlflow_logger.log_metrics({"final_val_loss": val_loss})

    return val_loss


if __name__ == "__main__":
    pl.seed_everything(42)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:", study.best_trial.params)
