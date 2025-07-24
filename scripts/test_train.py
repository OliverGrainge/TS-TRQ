from models import DiTModule 
from data import ImageNetDataModule
import pytorch_lightning as pl
from dotenv import load_dotenv

def main(): 
    dit = DiTModule() 
    datamodule = ImageNetDataModule(batch_size=2)
    datamodule.setup()

    trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
    trainer.fit(dit, datamodule)

if __name__ == "__main__":
    load_dotenv()
    main()