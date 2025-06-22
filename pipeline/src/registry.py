"""
Registry system for models and datamodules with automatic subgroup registration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type


class ModelRegistry:
    """Registry for models with their configurations."""

    _models: Dict[str, Tuple[Type, Type]] = {}  # name -> (model_class, config_class)

    @classmethod
    def register(cls, name: str, config_class: Type):
        """Decorator to register a model with its config."""

        def decorator(model_class: Type):
            cls._models[name] = (model_class, config_class)
            return model_class

        return decorator

    @classmethod
    def get_model_configs(cls):
        """Get all registered model configs for simple-parsing subgroups."""
        return {name: config_class for name, (_, config_class) in cls._models.items()}

    @classmethod
    def get_model(cls, name: str, config) -> Any:
        """Build model instance from config."""
        if name not in cls._models:
            raise ValueError(
                f"Model '{name}' not registered. Available: {list(cls._models.keys())}"
            )
        model_class, _ = cls._models[name]
        return model_class(config)

    @classmethod
    def get_registered_models(cls) -> dict[str, Tuple[Type, Type]]:
        """Get all registered models."""
        return cls._models.copy()


class DataModuleRegistry:
    """Registry for datamodules with their configurations."""

    _datamodules: Dict[
        str, Tuple[Type, Type]
    ] = {}  # name -> (datamodule_class, config_class)

    @classmethod
    def register(cls, name: str, config_class: Type):
        """Decorator to register a datamodule with its config."""

        def decorator(datamodule_class: Type):
            cls._datamodules[name] = (datamodule_class, config_class)
            return datamodule_class

        return decorator

    @classmethod
    def get_datamodule_configs(cls):
        """Get all registered datamodule configs for simple-parsing subgroups."""
        return {
            name: config_class for name, (_, config_class) in cls._datamodules.items()
        }

    @classmethod
    def get_datamodule(cls, name: str, config) -> Any:
        """Build datamodule instance from config."""
        if name not in cls._datamodules:
            raise ValueError(
                f"datamodule '{name}' not registered. Available: {list(cls._datamodules.keys())}"
            )
        datamodule_class, _ = cls._datamodules[name]
        return datamodule_class(config)

    @classmethod
    def get_registered_datamodules(cls) -> Dict[str, Tuple[Type, Type]]:
        """Get all registered datamodules."""
        return cls._datamodules.copy()


# Base classes for configs
@dataclass
class BaseModelConfig:
    """Base class for all model configurations."""

    __target__: str
    name: str


@dataclass
class BaseDataModuleConfig:
    """Base class for all datamodule configurations."""

    __target__: str
    name: str


# Convenience decorators
def register_model(name: str, config_class: Type):
    """Decorator to register a model with its config."""
    return ModelRegistry.register(name, config_class)


def register_datamodule(name: str, config_class: Type):
    """Decorator to register a datamodule with its config."""
    return DataModuleRegistry.register(name, config_class)


def get_subgroup_choices():
    """Get choices for simple-parsing subgroups."""
    return {
        "model": ModelRegistry.get_model_configs(),
        "datamodule": DataModuleRegistry.get_datamodule_configs(),
    }
