import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


@dataclass
class TrainerConfig:
    output_dir: Path
    max_epochs: int
    save_top_k: int = 0
    save_last: bool = False
    log_every_n_steps: int = 10
    patience: int = 10
    accumulate_grad_batches: int = 1


def setup_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )


def initialize_trainer(
    config: TrainerConfig,
    config_path: Path = Path("./params.yaml"),
):
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(Path(config.output_dir))
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    experiment_name = os.environ.get("DVC_EXP_NAME", None)  # if not dvc, than random

    wandb_logger = WandbLogger(
        project="deep-origin-task",
        name=experiment_name,
        log_model=False,
        config=config_dict,
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir=config.output_dir, name=experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir / "checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=config.save_top_k,
        save_last=config.save_last,
        save_weights_only=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config.patience, mode="min", verbose=True
    )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config.log_every_n_steps,
        reload_dataloaders_every_n_epochs=True,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    return trainer
