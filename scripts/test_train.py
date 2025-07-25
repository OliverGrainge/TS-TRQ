import pytorch_lightning as pl
from dotenv import load_dotenv

from data import ImageNetDataModule
from models import DiTModule


def main():
    dit = DiTModule(quant_type="tsvdlinear", quant_args={"rank": 192})
    print(dit.transformer)
    datamodule = ImageNetDataModule(batch_size=2)
    datamodule.setup()

    trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
    trainer.fit(dit, datamodule)


if __name__ == "__main__":
    load_dotenv()
    main()
