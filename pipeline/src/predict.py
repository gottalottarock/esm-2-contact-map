"""
Training script using simple-parsing subgroups with automatic registry.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# import seed to make all sampling deterministic
import utils.seed
from datamodules import *

# Import modules to register models and datamodules BEFORE using them
from models import *
from simple_parsing import ArgumentGenerationMode, ArgumentParser, NestedMode
from train import TrainConfig, init_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictConfig(TrainConfig):
    checkpoint_name: str = "best"
    output_predictions_path: Optional[str] = None


def find_best_checkpoint(checkpoint_dir: str | Path, metric: str, mode: str) -> Path:
    """Find the best checkpoint in the checkpoint directory."""

    checkpoint_files = [
        x for x in Path(checkpoint_dir).glob("*.ckpt") if metric.replace("/", "_") in x.name
    ]
    if len(checkpoint_files) == 0:
        raise ValueError("No checkpoint files found in the checkpoint directory")
    if mode == "max":
        file_name = max(checkpoint_files, key=lambda x: float(x.stem.split("=")[1]))
        logger.info(f"Found best checkpoint: {file_name}")
        return file_name
    elif mode == "min":
        file_name = min(checkpoint_files, key=lambda x: float(x.stem.split("=")[1]))
        logger.info(f"Found best checkpoint: {file_name}")
        return file_name
    else:
        raise ValueError(f"Invalid mode: {mode}")


def main(cfg: PredictConfig):
    model, datamodule, trainer = init_all(cfg)
    datamodule.setup("test")
    if cfg.checkpoint_name == "best":
        checkpoint_path = find_best_checkpoint(
            Path(cfg.trainer.output_dir) / "checkpoints", cfg.trainer.checkpoint.monitor, cfg.trainer.checkpoint.mode
        )
    else:
        checkpoint_path = Path(cfg.trainer.output_dir) / "checkpoints" / f"{cfg.checkpoint_name}.ckpt"
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)["state_dict"])
    predictions = trainer.predict(model, datamodule)
    if cfg.output_predictions_path is None:
        raise ValueError("output_predictions_path is required")
    with open(cfg.output_predictions_path, "wb") as f:
        pickle.dump(predictions, f)


if __name__ == "__main__":
    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.NESTED,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )
    parser.add_arguments(PredictConfig, dest="config")
    args = parser.parse_args()
    main(args.config)
