import lightning as L
import torch
from datasets import Dataset
from torch import Tensor
from torchmetrics.functional import accuracy

from few_shot.config import Config


class RAG(L.LightningModule):
    def __init__(
        self, cfg: Config, num_classes: int, train_db: Dataset
    ):  # , datamodule: DataModule
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.train_db = train_db
        self.num_cls = num_classes

    def forward(self, x: Tensor) -> Tensor:
        _, retrieved = self.train_db.get_nearest_examples_batch(
            "embed", x.cpu().numpy(), 1
        )
        return torch.cat([torch.tensor(data["label"]) for data in retrieved])

    def evaluate(self, batch, stage=None):
        _, y, emb = batch
        preds = self(emb)
        acc = accuracy(preds.cpu(), y.cpu(), "multiclass", num_classes=self.num_cls)
        if stage:
            self.log(f"{stage}_acc", acc, prog_bar=True)
            setattr(self, f"{stage}_acc", acc)
        return 0

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
