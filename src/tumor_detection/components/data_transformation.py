import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tumor_detection import logger
from tumor_detection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _load_images(self, folder: Path) -> list:
        images = []
        for img_path in sorted(folder.glob("*.jpg")):
            img = Image.open(img_path).convert("L").resize(
                (self.config.img_size, self.config.img_size)
            )
            images.append(np.array(img, dtype=np.float32) / 255.0)
        return images

    def transform(self):
        healthy_imgs = self._load_images(self.config.data_path / "Healthy")
        tumor_imgs = self._load_images(self.config.data_path / "Tumor")

        healthy_train, healthy_test = train_test_split(
            healthy_imgs, test_size=0.2, random_state=42
        )
        _, tumor_test = train_test_split(
            tumor_imgs, test_size=0.2, random_state=42
        )

        X_train = np.array(healthy_train)
        y_train = np.zeros(len(X_train), dtype=np.int32)

        X_test = np.array(healthy_test + tumor_test)
        y_test = np.concatenate([
            np.zeros(len(healthy_test), dtype=np.int32),
            np.ones(len(tumor_test), dtype=np.int32),
        ])

        np.save(self.config.root_dir / "X_train.npy", X_train)
        np.save(self.config.root_dir / "y_train.npy", y_train)
        np.save(self.config.root_dir / "X_test.npy", X_test)
        np.save(self.config.root_dir / "y_test.npy", y_test)

        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
        logger.info(f"Test label distribution — healthy: {(y_test == 0).sum()}, tumor: {(y_test == 1).sum()}")
