from lightning.pytorch.cli import LightningCLI

from .data import FPHADataModule

LightningCLI(datamodule_class=FPHADataModule)
