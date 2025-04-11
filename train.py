from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer, seed_everything
from utils.config import get_train_config_obj
from utils.common import setup_callbacks
from dataset.dataloader import TrajectoryDataloaderModule
from model.trajectory_generator_train import TrajectoryTrainingModule
import argparse


def main(config_obj):
    seed_everything(42)

    model = TrajectoryTrainingModule(config_obj)

    print(model.gen_model)

    data = TrajectoryDataloaderModule(cfg=config_obj)

    run_name = (
        f"Emb={config_obj.embedding_dim}, Dim={config_obj.tf_de_dim}, "
        f"Head={config_obj.tf_de_heads}, Layer={config_obj.tf_de_layers}"
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="e2e_planner",
        run_name=run_name,
        tracking_uri="http://172.21.191.16:9999"
    )

    trainer = Trainer(
        callbacks=setup_callbacks(config_obj, "val_accuracy", "max"),
        logger=[mlflow_logger],
        accelerator="gpu",
        # strategy='ddp_find_unused_parameters_true',
        devices=config_obj.num_gpus,
        max_epochs=config_obj.epochs,
        log_every_n_steps=config_obj.log_every_n_steps,
        check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
        profiler='simple'
    )

    trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for trajectory generator.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training.yaml",
        help="Path to the training config file."
    )
    args = parser.parse_args()
    cfg_path = args.config
    train_config_obj = get_train_config_obj(config_path=cfg_path)
    main(train_config_obj)
