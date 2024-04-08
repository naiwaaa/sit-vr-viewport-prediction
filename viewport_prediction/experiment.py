from __future__ import annotations

from typing import TYPE_CHECKING

import wandb
from viewport_prediction.utils import console
from viewport_prediction.config import ExperimentConfig
from viewport_prediction.models import ALL_MODELS
from viewport_prediction.entities import Session


if TYPE_CHECKING:
    from pathlib import Path

    from viewport_prediction.types import DataLoader
    from viewport_prediction.models import BaseModel
    from viewport_prediction.data.base_dataset import BaseDataset
    from viewport_prediction.config.experiment_config import BaseModelConfig


def run_experiment(model_name: str, config_file: Path) -> None:
    model_cls = ALL_MODELS[model_name]
    config_cls = model_cls.Config
    dataset_cls = model_cls.Dataset

    config = ExperimentConfig[config_cls].read_from_file(config_file)  # type: ignore

    console.print_divider("Init wandb")
    wandb_session = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        resume=config.wandb.resume,
        config=config.dict(),
    )

    console.print_divider("Experiment config")
    console.print_dict(config.dict())

    console.print_divider("Prepare dataset")
    train_dataloader = _create_dataloader(
        config.data.train_video_indices,
        config,
        dataset_cls,
    )
    val_dataloader = _create_dataloader(
        config.data.val_video_indices,
        config,
        dataset_cls,
    )
    test_dataloader = _create_dataloader(
        config.data.test_video_indices,
        config,
        dataset_cls,
    )

    with wandb_session:
        model = model_cls(config)
        _run_pipeline(model, train_dataloader, val_dataloader, test_dataloader)

        console.print_divider("Clean up")


def _run_pipeline(
    model: BaseModel[BaseModelConfig],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    test_dataloader: DataLoader,
) -> None:
    """Run the pipeline including building, training and testing the model."""
    console.print_divider("Build model architecture")
    model.build()
    model.summary()
    wandb.log({"model_architecture": wandb.Image(model.plot_model(), mode="RGB")})

    console.print_divider("Train model")
    callbacks = [
        wandb.keras.WandbCallback(
            monitor="loss" if val_dataloader is None else "val_loss",
            mode="min",
            save_model=True,
            save_graph=True,
            save_weights_only=True,
            log_weights=True,
            log_gradients=False,
        ),
    ]
    model.fit(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callbacks=callbacks,
    )

    console.print_divider("Evaluate model")
    result = model.evaluate(test_dataloader)
    console.print("Evaluation result:")
    console.print_dict(result)
    wandb.log(
        {
            "evaluation_result": wandb.Table(  # type: ignore
                columns=list(result.keys()),
                data=[list(result.values())],
            ),
        },
    )


def _create_dataloader(
    video_indices: list[int] | None,
    config: ExperimentConfig[BaseModelConfig],
    dataset_type: type[BaseDataset],
) -> DataLoader | None:
    if video_indices is None:
        return None

    sessions = []
    for video_id in video_indices:
        for session_dir in config.data.data_dir.glob(f"video_{video_id:02}_user_*"):
            sessions.append(Session(session_dir))

    return dataset_type(
        sessions,
        past_window_size=config.model.past_window_size,
        future_window_size=config.model.future_window_size,
    ).loader(config.training.batch_size)
