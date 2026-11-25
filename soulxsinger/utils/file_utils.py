"""Configuration loading for SVC inference."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    config = OmegaConf.load(config_path)
    if config.get("base_config") is not None:
        config = OmegaConf.merge(OmegaConf.load(config.base_config), config)
    return config
