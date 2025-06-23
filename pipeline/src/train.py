"""
Training script using simple-parsing subgroups with automatic registry.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from pprint import pformat
import lightning as L

# import seed to make all sampling deterministic
import utils.seed

from simple_parsing import ArgumentParser, subgroups, ArgumentGenerationMode, NestedMode

from registry import (
    ModelRegistry,
    DataModuleRegistry,
    BaseModelConfig,
    BaseDataModuleConfig,
)
from trainers.trainer import TrainerConfig, initialize_trainer

# Import modules to register models and datamodules BEFORE using them
from models import *
from datamodules import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Trainer config
    trainer: TrainerConfig

    # Which model to use
    model: BaseModelConfig = subgroups(
        ModelRegistry.get_model_configs(),  # type: ignore
    )

    # Which datamodule to use
    datamodule: BaseDataModuleConfig = subgroups(
        DataModuleRegistry.get_datamodule_configs(),  # type: ignore
    )

def init_all(cfg: TrainConfig) -> tuple[L.LightningModule, L.LightningDataModule, L.Trainer]:
    datamodule = DataModuleRegistry.get_datamodule(
        cfg.datamodule._target_, cfg.datamodule
    )
    model = ModelRegistry.get_model(cfg.model._target_, cfg.model)
    trainer = initialize_trainer(cfg.trainer)

    return model, datamodule, trainer

def main(cfg: TrainConfig):
    model, datamodule, trainer = init_all(cfg)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.NESTED,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )
    parser.add_arguments(TrainConfig, dest="config")
    args = parser.parse_args()
    main(args.config)
