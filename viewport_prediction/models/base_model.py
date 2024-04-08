from __future__ import annotations

from typing import TYPE_CHECKING, Generic

import tempfile
from pathlib import Path

import numpy as np

from tensorflow import keras as tfk

from viewport_prediction.utils import image, console
from viewport_prediction.metrics.rmse import RMSEMetric
from viewport_prediction.data.base_dataset import BaseDataset
from viewport_prediction.config.experiment_config import ModelConfigT
from viewport_prediction.metrics.orthodromic_distance import OrthodromicDistanceMetric


if TYPE_CHECKING:
    from viewport_prediction.types import (
        BatchData,
        DataLoader,
        NDArrayInt,
        NDArrayFloat,
    )
    from viewport_prediction.config.experiment_config import ExperimentConfig


class BaseModel(Generic[ModelConfigT]):
    model: tfk.Model
    Config: type[ModelConfigT]  # pylint: disable=invalid-name
    Dataset: type[BaseDataset]  # pylint: disable=invalid-name

    def __init__(self, config: ExperimentConfig[ModelConfigT]) -> None:
        self.config = config
        self.history: tfk.callbacks.History | None = None

    def build(self) -> None:
        raise NotImplementedError()

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        callbacks: list[tfk.callbacks.Callback],
    ) -> None:
        loss_func = self.config.training.loss_func
        optimizer = tfk.optimizers.Adam(learning_rate=self.config.training.learning_rate)
        callbacks = [
            tfk.callbacks.EarlyStopping(
                monitor="loss" if val_dataloader is None else "val_loss",
                mode="min",
                patience=10,
                min_delta=1e-6,
                restore_best_weights=True,
            ),
            *callbacks,
        ]

        self.model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=[],
        )

        self.history = self.model.fit(
            train_dataloader,
            validation_data=val_dataloader,
            epochs=self.config.training.max_epochs,
            callbacks=callbacks,
            verbose="auto",
        )

    def predict(self, x: BatchData) -> NDArrayFloat:
        pred: NDArrayFloat = self.model.predict(x)
        return pred * [np.pi, 2 * np.pi]

    def evaluate(self, test_dataloader: DataLoader) -> dict[str, float]:
        metric_rmse_inclination = RMSEMetric()
        metric_rmse_azimuth = RMSEMetric()
        metric_orthodromic_distance = OrthodromicDistanceMetric()

        for batch in test_dataloader:
            *x, y = batch
            pred = self.model.predict(x)

            metric_rmse_inclination.update(pred[:, :, 0], y.numpy()[:, :, 0])
            metric_rmse_azimuth.update(pred[:, :, 1], y.numpy()[:, :, 1])
            metric_orthodromic_distance.update(pred, y.numpy())

        return {
            "rmse_inclination": metric_rmse_inclination.compute(),
            "rmse_azimuth": metric_rmse_azimuth.compute(),
            "orthodromic_distance_mean": metric_orthodromic_distance.compute()["mean"],
            "orthodromic_distance_std": metric_orthodromic_distance.compute()["std"],
        }

    def plot_model(self, out_file: str | Path | None = None) -> NDArrayInt:
        file = Path(tempfile.mkstemp(suffix=".png")[1] if out_file is None else out_file)

        tfk.utils.plot_model(
            self.model,
            to_file=file,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=False,
            show_layer_activations=True,
        )

        img_data = image.imread(file, color_mode="rgb")

        if out_file is None:
            file.unlink(missing_ok=True)

        return img_data.astype(int)

    def save(self, out_dir: Path) -> None:
        self.model.save(out_dir, save_format="tf")

    def summary(self) -> None:
        self.model.summary(
            print_fn=console.print,
            expand_nested=True,
            show_trainable=True,
        )
