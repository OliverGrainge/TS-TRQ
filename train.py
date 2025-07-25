import os

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config_utils import get_train_config, Config

load_dotenv()
torch.set_float32_matmul_precision("high")


def get_module(config, kwargs, checkpoint_path):
    if config.module == "dit":
        from models import DiTModule

        if checkpoint_path is not None:
            return DiTModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return DiTModule(**kwargs)
    elif config.module == "mlp":
        from models import MLPModule

        if checkpoint_path is not None:
            return MLPModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return MLPModule(**kwargs)
    elif config.module == "vit":
        from models import ViTModule

        if checkpoint_path is not None:
            return ViTModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return ViTModule(**kwargs)
    else:
        raise ValueError(f"Invalid module: {config.module}")


def get_datamodule(config, kwargs):
    if config.dataset == "imagenet":
        from data import ImageNetDataModule

        return ImageNetDataModule(**kwargs)
    elif config.dataset == "cifar100":
        from data import CIFAR100DataModule

        return CIFAR100DataModule(**kwargs)


def main():
    # Load configuration from YAML file
    config_dict = get_train_config()
    config = Config(config_dict)
    
    print(f"Training configuration: {config}")
    
    module_kwargs = {"learning_rate": config.learning_rate, "reg_scale": config.reg_scale}
    quant_kwargs = {
        "rank": config.quant_rank,
    }
    datamodule_kwargs = {"batch_size": config.batch_size}

    # Load the model module
    module = get_module(config, module_kwargs, config.checkpoint)

    # Quantize the module
    if config.quant_type != "none":
        module.apply_quantization(quant_type=config.quant_type, quant_kwargs=quant_kwargs)

    print(module)

    # Load the Data Module
    datamodule = get_datamodule(config, datamodule_kwargs)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    wandb_logger = WandbLogger(
        project=config.project + "_" + config.module, log_model=False
    )

    # Use pathlib for easier and more robust path handling
    from pathlib import Path

    # Determine checkpoint directory
    checkpoint_dir = Path(config.save_dir) / config.module / config.quant_type

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(checkpoint_dir),
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=5,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        gradient_clip_val=1.0,
        enable_progress_bar=config.enable_progress_bar,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
