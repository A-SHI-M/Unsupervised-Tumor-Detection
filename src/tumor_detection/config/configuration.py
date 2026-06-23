from tumor_detection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from tumor_detection.utils import read_yaml, create_directories
from tumor_detection.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainingConfig
from pathlib import Path


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            img_size=config.img_size,
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params
        create_directories([config.root_dir, "trainedmodels"])
        return ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            epochs=params.EPOCHS,
            pretrain_epochs=params.PRETRAIN_EPOCHS,
            batch_size=params.BATCH_SIZE,
            img_size=params.IMG_SIZE,
            latent_dim=params.LATENT_DIM,
            lr_discriminator=params.LR_DISCRIMINATOR,
            lr_bigan=params.LR_BIGAN,
            lr_reconstruction=params.LR_RECONSTRUCTION,
        )
