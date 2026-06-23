from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    img_size: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    epochs: int
    pretrain_epochs: int
    batch_size: int
    img_size: int
    latent_dim: int
    lr_discriminator: float
    lr_bigan: float
    lr_reconstruction: float