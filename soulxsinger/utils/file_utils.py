"""Configuration loading for SVC inference."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)
