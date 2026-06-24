import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from tumor_detection import logger
from tumor_detection.entity.config_entity import ModelEvaluationConfig
from tumor_detection.utils import save_json


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.img_shape = (config.img_size, config.img_size, 1)

    def _load_models(self):
        encoder = tf.keras.models.load_model(self.config.model_path / "encoder.keras")
        generator = tf.keras.models.load_model(self.config.model_path / "generator.keras")
        logger.info("Loaded encoder and generator from trainedmodels/")
        return encoder, generator

    def _load_test_data(self):
        X_test = np.load(self.config.test_data_path / "X_test.npy")[..., np.newaxis].astype(np.float32)
        y_test = np.load(self.config.test_data_path / "y_test.npy")
        logger.info(f"Test set: {X_test.shape}, labels: {y_test.shape} (healthy={( y_test==0).sum()}, tumor={(y_test==1).sum()})")
        return X_test, y_test

    def _calculate_anomaly_scores(self, X_test, encoder, generator):
        encoded_z = encoder.predict(X_test, verbose=0)
        reconstructed = generator.predict(encoded_z, verbose=0)

        mse_errors = np.mean(np.square(X_test - reconstructed), axis=(1, 2, 3))

        ssim_errors = []
        for i in range(len(X_test)):
            ssim_val = tf.image.ssim(
                tf.convert_to_tensor(X_test[i:i+1]),
                tf.convert_to_tensor(reconstructed[i:i+1]),
                max_val=1.0,
            ).numpy()[0]
            ssim_errors.append(1.0 - ssim_val)

        ssim_errors = np.array(ssim_errors)
        anomaly_scores = 0.7 * mse_errors + 0.3 * ssim_errors
        logger.info(f"Anomaly scores — min: {anomaly_scores.min():.4f}, max: {anomaly_scores.max():.4f}")
        return anomaly_scores, reconstructed

    def _compute_metrics(self, anomaly_scores, y_test):
        fpr, tpr, thresholds = roc_curve(y_test, anomaly_scores)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, anomaly_scores)
        avg_precision = average_precision_score(y_test, anomaly_scores)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = float(thresholds[optimal_idx])

        predictions = (anomaly_scores >= optimal_threshold).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        specificity = tn / (tn + fp)

        logger.info("\n===== ANOMALY DETECTION PERFORMANCE =====")
        logger.info(f"ROC AUC:            {roc_auc:.4f}")
        logger.info(f"Average Precision:  {avg_precision:.4f}")
        logger.info(f"Optimal Threshold:  {optimal_threshold:.4f}")
        logger.info(f"Accuracy:           {accuracy:.4f}")
        logger.info(f"Precision:          {precision:.4f}")
        logger.info(f"Recall (Sens.):     {recall:.4f}")
        logger.info(f"Specificity:        {specificity:.4f}")
        logger.info(f"F1 Score:           {f1:.4f}")
        logger.info(f"Confusion Matrix:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        logger.info("=========================================\n")

        return {
            "roc_auc": roc_auc, "average_precision": avg_precision,
            "optimal_threshold": optimal_threshold,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "specificity": specificity, "f1_score": f1,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }, fpr, tpr, precision_curve, recall_curve, predictions

    def _plot_evaluation_metrics(self, fpr, tpr, roc_auc, precision_curve, recall_curve,
                                  avg_precision, anomaly_scores, y_test,
                                  optimal_threshold, predictions):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0]); axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate'); axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('Receiver Operating Characteristic (ROC)')
        axes[0, 0].legend(loc='lower right'); axes[0, 0].grid(True, linestyle='--', alpha=0.7)

        axes[0, 1].plot(recall_curve, precision_curve, color='blue', lw=2, label=f'AP = {avg_precision:.3f}')
        axes[0, 1].set_xlabel('Recall'); axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc='upper right'); axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].set_ylim([0.0, 1.05]); axes[0, 1].set_xlim([0.0, 1.0])

        axes[1, 0].hist(anomaly_scores[y_test == 0], bins=30, alpha=0.5, label='Normal', color='green')
        axes[1, 0].hist(anomaly_scores[y_test == 1], bins=30, alpha=0.5, label='Abnormal', color='red')
        axes[1, 0].axvline(x=optimal_threshold, color='black', linestyle='--',
                           label=f'Threshold: {optimal_threshold:.3f}')
        axes[1, 0].set_xlabel('Anomaly Score'); axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Anomaly Scores')
        axes[1, 0].legend(); axes[1, 0].grid(True, linestyle='--', alpha=0.7)

        cm = confusion_matrix(y_test, predictions)
        axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title('Confusion Matrix')
        fig.colorbar(axes[1, 1].images[0], ax=axes[1, 1])
        tick_marks = np.arange(2)
        axes[1, 1].set_xticks(tick_marks); axes[1, 1].set_xticklabels(['Normal', 'Abnormal'], rotation=45)
        axes[1, 1].set_yticks(tick_marks); axes[1, 1].set_yticklabels(['Normal', 'Abnormal'])
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 1].text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                                color='white' if cm[i, j] > thresh else 'black')
        axes[1, 1].set_ylabel('True Label'); axes[1, 1].set_xlabel('Predicted Label')

        plt.tight_layout()
        path = str(self.config.root_dir / "evaluation_metrics.png")
        plt.savefig(path); plt.close(fig)
        return path

    def _plot_metrics_summary(self, metrics):
        names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'ROC AUC']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                  metrics['f1_score'], metrics['specificity'], metrics['roc_auc']]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, values, color=['blue', 'green', 'red', 'purple', 'orange', 'teal'])
        ax.set_ylim([0, 1.1]); ax.set_ylabel('Score')
        ax.set_title('Anomaly Detection Performance Metrics')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}', ha='center')
        plt.tight_layout()
        path = str(self.config.root_dir / "metrics_summary.png")
        plt.savefig(path); plt.close(fig)
        return path

    def _plot_reconstructions(self, X_test, y_test, reconstructed, anomaly_scores, label, filename):
        indices = np.where(y_test == label)[0]
        np.random.shuffle(indices)
        examples = indices[:5]
        label_name = "Normal" if label == 0 else "Abnormal"

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, idx in enumerate(examples):
            axes[0, i].imshow(X_test[idx, :, :, 0], cmap='gray')
            axes[0, i].set_title(f"{label_name}\nScore: {anomaly_scores[idx]:.4f}")
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed[idx, :, :, 0], cmap='gray')
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')
        plt.tight_layout()
        path = str(self.config.root_dir / filename)
        plt.savefig(path); plt.close(fig)
        return path

    def evaluate(self):
        encoder, generator = self._load_models()
        X_test, y_test = self._load_test_data()
        anomaly_scores, reconstructed = self._calculate_anomaly_scores(X_test, encoder, generator)

        metrics, fpr, tpr, precision_curve, recall_curve, predictions = self._compute_metrics(
            anomaly_scores, y_test
        )

        eval_plot = self._plot_evaluation_metrics(
            fpr, tpr, metrics['roc_auc'],
            precision_curve, recall_curve, metrics['average_precision'],
            anomaly_scores, y_test, metrics['optimal_threshold'], predictions,
        )
        summary_plot = self._plot_metrics_summary(metrics)
        normal_plot = self._plot_reconstructions(
            X_test, y_test, reconstructed, anomaly_scores, label=0, filename="normal_reconstructions.png"
        )
        abnormal_plot = self._plot_reconstructions(
            X_test, y_test, reconstructed, anomaly_scores, label=1, filename="abnormal_reconstructions.png"
        )

        metrics_path = self.config.root_dir / "metrics.json"
        save_json(metrics_path, metrics)
        logger.info(f"Metrics saved to {metrics_path}")

        load_dotenv()
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        run_id_file = self.config.model_path / "mlflow_run_id.txt"
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
            with mlflow.start_run(run_id=run_id):
                mlflow.log_params({"optimal_threshold": metrics['optimal_threshold']})
                mlflow.log_metrics({
                    "roc_auc": metrics['roc_auc'],
                    "average_precision": metrics['average_precision'],
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1_score": metrics['f1_score'],
                    "specificity": metrics['specificity'],
                })
                for artifact in [eval_plot, summary_plot, normal_plot, abnormal_plot, str(metrics_path)]:
                    mlflow.log_artifact(artifact)
            logger.info("Evaluation metrics and plots logged to existing MLflow run.")
        else:
            logger.info("No MLflow run ID found — skipping MLflow logging (run stage 03 first).")
