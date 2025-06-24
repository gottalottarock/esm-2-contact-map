import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

@dataclass
class CheckpointConfig:
    monitor: str
    mode: str
    save_top_k: int = 1
    save_last: bool = False

@dataclass
class EarlyStoppingConfig:
    monitor: str
    patience: int
    mode: str

@dataclass
class TrainerConfig:
    output_dir: Path
    max_epochs: int
    log_every_n_steps: int
    accumulate_grad_batches: int
    checkpoint: CheckpointConfig
    early_stopping: EarlyStoppingConfig
    val_check_interval: int = 100


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
    params_path: Path = Path("./params.yaml"),
):
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(Path(config.output_dir))
    with open(params_path, "r") as file:
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
        filename="{epoch}-{step}-%s={%s:.4f}" % (config.checkpoint.monitor.replace("/", "_"), config.checkpoint.monitor),
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k,
        save_last=config.checkpoint.save_last,
        save_weights_only=True,
        auto_insert_metric_name=False
    )

    early_stopping = EarlyStopping(
        monitor=config.early_stopping.monitor,
        patience=config.early_stopping.patience,
        mode=config.early_stopping.mode,
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        reload_dataloaders_every_n_epochs=True,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    return trainer
