from dataclasses import dataclass


# ! BASE
# region
@dataclass
class Trainer:
    dropout: float
    epoch: int

    lr: float
    swa: float | None
    decay: float


@dataclass
class DataLoader:
    name: str
    shot: int
    augmentation: list

    batch_size: int


# endregion


@dataclass
class Config:
    base_model: str
    training_type: str
    version: str
    trainer: Trainer
    dataloader: DataLoader
