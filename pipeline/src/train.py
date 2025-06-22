"""
Training script using simple-parsing subgroups with automatic registry.
"""

import os
from dataclasses import dataclass
from typing import Optional

from simple_parsing import ArgumentParser, subgroups, ArgumentGenerationMode, NestedMode

from registry import ModelRegistry, DataModuleRegistry
from trainers.trainer import TrainerConfig, initialize_trainer

# Import modules to register models and datamodules BEFORE using them
from models import *
from datamodules import *


@dataclass
class Config:
    # Trainer config
    trainer: TrainerConfig

    # Which model to use
    model = subgroups(
        ModelRegistry.get_model_configs(),  # type: ignore
    )

    # Which datamodule to use
    datamodule = subgroups(
        DataModuleRegistry.get_datamodule_configs(),  # type: ignore
    )


def main(cfg: Config):
    datamodule = DataModuleRegistry.get_datamodule(
        cfg.datamodule.__target__, cfg.datamodule
    )
    model = ModelRegistry.get_model(cfg.model.__target__, cfg.model)
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
