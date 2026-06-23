import sys
import shutil
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

sys.path.insert(0, str(Path(__file__).parents[3]))
from model.BIGAN import (
    build_encoder, build_generator, build_discriminator,
    build_bigan, build_perceptual_model, build_reconstruction_model,
    ssim_loss,
)

from tumor_detection import logger
from tumor_detection.entity.config_entity import ModelTrainingConfig
from tumor_detection.utils import create_directories


class BiGANTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.img_shape = (config.img_size, config.img_size, 1)

    def _prepare_data(self) -> np.ndarray:
        X = np.load(self.config.train_data_path)
        return X[..., np.newaxis].astype(np.float32)

    def _save_progress_images(self, x_train, encoder, generator, filepath: str):
        idx = np.random.randint(0, x_train.shape[0], 5)
        samples = x_train[idx]
        recon = generator.predict(encoder.predict(samples, verbose=0), verbose=0)

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            axes[0, i].imshow(samples[i, :, :, 0], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title("Original")
            axes[1, i].imshow(recon[i, :, :, 0], cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title("Reconstructed")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close(fig)

    def _combined_loss(self, perceptual_model=None):
        def with_perceptual(y_true, y_pred):
            y_true_rgb = tf.concat([y_true, y_true, y_true], axis=-1)
            y_pred_rgb = tf.concat([y_pred, y_pred, y_pred], axis=-1)
            y_true_rgb = preprocess_input(y_true_rgb * 255.0)
            y_pred_rgb = preprocess_input(y_pred_rgb * 255.0)
            true_feats = perceptual_model(y_true_rgb)
            pred_feats = perceptual_model(y_pred_rgb)
            perc = sum(tf.reduce_mean(tf.square(t - p)) for t, p in zip(true_feats, pred_feats))
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            ssim = ssim_loss(y_true, y_pred)
            return 0.5 * mse + 0.2 * mae + 0.2 * perc + 0.1 * ssim

        def without_perceptual(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            ssim = ssim_loss(y_true, y_pred)
            return 0.6 * mse + 0.3 * mae + 0.1 * ssim

        return with_perceptual if perceptual_model is not None else without_perceptual

    def train(self):
        mlflow.set_tracking_uri("mlruns")
        x_train = self._prepare_data()
        progress_dir = self.config.root_dir / "progress_images"
        create_directories([self.config.root_dir, progress_dir])

        # Build models
        encoder = build_encoder(self.img_shape, self.config.latent_dim)
        generator = build_generator(self.config.latent_dim, self.img_shape)
        discriminator = build_discriminator(self.img_shape, self.config.latent_dim)
        discriminator.compile(
            optimizer=Adam(self.config.lr_discriminator, beta_1=0.5),
            loss='binary_crossentropy',
        )
        bigan = build_bigan(
            generator, discriminator, encoder,
            self.config.latent_dim, self.img_shape, self.config.lr_bigan,
        )

        try:
            perceptual_model = build_perceptual_model(self.img_shape)
            logger.info("Using perceptual loss with VGG16")
        except Exception:
            perceptual_model = None
            logger.info("VGG16 unavailable — using standard combined loss")

        recon_model = build_reconstruction_model(
            encoder, generator, self.img_shape, self.config.lr_reconstruction
        )
        recon_model.compile(
            optimizer=Adam(self.config.lr_reconstruction, beta_1=0.5),
            loss=self._combined_loss(perceptual_model),
        )

        ones = np.ones((self.config.batch_size, 1))
        zeros = np.zeros((self.config.batch_size, 1))

        pretrain_losses, d_losses, g_losses, recon_losses = [], [], [], []

        with mlflow.start_run():
            mlflow.log_params({
                "epochs": self.config.epochs,
                "pretrain_epochs": self.config.pretrain_epochs,
                "batch_size": self.config.batch_size,
                "img_size": self.config.img_size,
                "latent_dim": self.config.latent_dim,
                "lr_discriminator": self.config.lr_discriminator,
                "lr_bigan": self.config.lr_bigan,
                "lr_reconstruction": self.config.lr_reconstruction,
                "perceptual_loss": perceptual_model is not None,
            })

            # Phase 1: Pre-train reconstruction
            logger.info("Phase 1: Pre-training reconstruction model...")
            for epoch in range(self.config.pretrain_epochs):
                idx = np.random.randint(0, x_train.shape[0], self.config.batch_size)
                r_loss = float(recon_model.train_on_batch(x_train[idx], x_train[idx]))
                pretrain_losses.append(r_loss)

                if epoch % 10 == 0:
                    logger.info(f"Pre-train {epoch}/{self.config.pretrain_epochs} — recon_loss: {r_loss:.4f}")
                    mlflow.log_metric("pretrain_recon_loss", r_loss, step=epoch)

                if epoch % 20 == 0:
                    img_path = str(progress_dir / f"pretraining_epoch_{epoch}.png")
                    self._save_progress_images(x_train, encoder, generator, img_path)
                    mlflow.log_artifact(img_path)

            # Phase 2: Joint training
            logger.info("Phase 2: Joint training...")
            d_loss = 0.0
            for epoch in range(self.config.epochs):
                idx = np.random.randint(0, x_train.shape[0], self.config.batch_size)
                imgs = x_train[idx]
                z = np.random.normal(0, 1, (self.config.batch_size, self.config.latent_dim))

                if epoch % 5 == 0:
                    real_z = encoder.predict(imgs, verbose=0)
                    fake_imgs = generator.predict(z, verbose=0)
                    d_real = float(discriminator.train_on_batch([imgs, real_z], ones))
                    d_fake = float(discriminator.train_on_batch([fake_imgs, z], zeros))
                    d_loss = 0.5 * (d_real + d_fake)

                g_loss = float(bigan.train_on_batch([z, imgs], [ones, zeros])[0])

                r_loss = 0.0
                for _ in range(3):
                    idx2 = np.random.randint(0, x_train.shape[0], self.config.batch_size)
                    r_loss = float(recon_model.train_on_batch(x_train[idx2], x_train[idx2]))

                d_losses.append(d_loss)
                g_losses.append(g_loss)
                recon_losses.append(r_loss)

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{self.config.epochs} — "
                        f"D: {d_loss:.4f}  G: {g_loss:.4f}  R: {r_loss:.4f}"
                    )
                    mlflow.log_metrics(
                        {"d_loss": d_loss, "g_loss": g_loss, "recon_loss": r_loss},
                        step=epoch,
                    )

                if epoch % 50 == 0:
                    img_path = str(progress_dir / f"joint_training_epoch_{epoch}.png")
                    self._save_progress_images(x_train, encoder, generator, img_path)
                    mlflow.log_artifact(img_path)

            # Save models to artifacts dir and trainedmodels dir
            trained_dir = Path("trainedmodels")
            for name, model in [
                ("encoder", encoder),
                ("generator", generator),
                ("discriminator", discriminator),
            ]:
                artifact_path = str(self.config.root_dir / f"{name}.keras")
                trained_path = str(trained_dir / f"{name}.keras")
                model.save(artifact_path)
                shutil.copy(artifact_path, trained_path)
                mlflow.log_artifact(artifact_path)
                logger.info(f"Saved {name} -> {artifact_path} and {trained_path}")

            # Plot loss curves
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(pretrain_losses)
            axes[0].set_title("Phase 1: Reconstruction Pre-training")
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

            axes[1].plot(d_losses, label="D Loss")
            axes[1].set_title("Discriminator Loss")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()

            axes[2].plot(g_losses, label="G/E Loss", color="orange")
            axes[2].plot(recon_losses, label="Recon Loss", color="green")
            axes[2].set_title("Phase 2: Joint Training")
            axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss"); axes[2].legend()

            plt.tight_layout()
            loss_path = str(self.config.root_dir / "loss_curves.png")
            plt.savefig(loss_path)
            plt.close(fig)
            mlflow.log_artifact(loss_path)
            logger.info("Training complete. All artifacts saved.")
