"""
Training script using simple-parsing subgroups with automatic registry.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

# import seed to make all sampling deterministic
import utils.seed

from simple_parsing import ArgumentParser, subgroups, ArgumentGenerationMode, NestedMode

from registry import ModelRegistry, DataModuleRegistry, BaseModelConfig, BaseDataModuleConfig
from trainers.trainer import TrainerConfig, initialize_trainer

# Import modules to register models and datamodules BEFORE using them
from models import *
from datamodules import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
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


def main(cfg: Config):
    logger.info(f"Using model: {cfg.model._target_}")
    logger.info(f"Using datamodule: {cfg.datamodule._target_}")

    logger.info(f"Using datamodule config: {cfg.datamodule}")
    datamodule = DataModuleRegistry.get_datamodule(
        cfg.datamodule._target_, cfg.datamodule
    )
    logger.info(f"Using model config: {cfg.model}")
    model = ModelRegistry.get_model(cfg.model._target_, cfg.model)
    logger.info(f"Using trainer config: {cfg.trainer}")
    trainer = initialize_trainer(cfg.trainer)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.NESTED,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    main(args.config)
