
import argparse
import pytorch_lightning as pl
from dotenv import load_dotenv
import torch 

load_dotenv()
torch.set_float32_matmul_precision("high")

from models import DiTModule 
from data import ImageNetDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT on ImageNet with PyTorch Lightning")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=int, default=32, help="Precision (16, 32, etc.)")
    parser.add_argument("--project", type=str, default="dit-imagenet", help="wandb project name")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--reg_scale", type=float, default=0.5, help="Regularization scale")
    parser.add_argument("--quant_type", type=str, default="tsvdlinear", help="Quantization type")
    parser.add_argument("--quant_rank", type=int, default=192, help="Quantization rank")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--devices", type=str, default="1", help="Number of GPUs or comma separated list of GPU ids (e.g., '0,1')")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Training accelerator: 'cpu', 'gpu', etc.")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy: 'ddp', 'dp', etc.")
    return parser.parse_args()

def main():
    args = parse_args()
    dit = DiTModule(
        learning_rate=args.learning_rate,
        quant_type=args.quant_type,
        quant_args={"rank": args.quant_rank},
        reg_scale=args.reg_scale
    )
    datamodule = ImageNetDataModule(batch_size=args.batch_size)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = WandbLogger(project=args.project, log_model=False)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_dir,
        filename="dit-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=5,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        gradient_clip_val=1.0,
    )
    trainer.fit(dit, train_loader, val_loader)

if __name__ == "__main__":
    main()