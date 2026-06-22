import os
import re
import zipfile
import gdown
from pathlib import Path
from tumor_detection import logger
from tumor_detection.entity.config_entity import DataIngestionConfig
from tumor_detection.utils import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def _gdrive_download_url(sharing_url: str) -> str:
        match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", sharing_url)
        if match:
            return f"https://drive.google.com/uc?id={match.group(1)}"
        return sharing_url

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            url = self._gdrive_download_url(self.config.source_URL)
            gdown.download(url, str(self.config.local_data_file))
            logger.info(
                f"Downloaded: {self.config.local_data_file} "
                f"({get_size(Path(self.config.local_data_file))})"
            )
        else:
            logger.info(
                f"File already exists: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info(f"Extracted zip file into: {self.config.unzip_dir}")
        os.remove(self.config.local_data_file)
        logger.info(f"Deleted zip file: {self.config.local_data_file}")
