from models import DiTModule 
from data import ImageNetDataModule
import pytorch_lightning as pl
from dotenv import load_dotenv

def main(): 
    dit = DiTModule(quant_type="nbitlinear", quant_args={"n_bits": 8}) 
    datamodule = ImageNetDataModule(batch_size=4)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    # Set up Wandb logger
    wandb_logger = WandbLogger(project="dit-imagenet", log_model=False)

    # Set up model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="dit-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=100,
        precision=32,
        accumulate_grad_batches=4,
    )
    trainer.fit(dit, train_loader, val_loader)

if __name__ == "__main__":
    load_dotenv()
    main()