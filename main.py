import csv
from sys import settrace
from typing import cast

import hydra
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging
from torch import Tensor, nn
from torchmetrics.functional import accuracy

from few_shot.config import Config
from few_shot.dataset import DataModule

settrace


class LinearProbing(L.LightningModule):
    def __init__(
        self, cfg: Config, input_dim: int, num_cls: int
    ):  # , datamodule: DataModule
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_cls),
        )
        self.cfg = cfg
        self.lr = cfg.trainer.lr
        self.num_cls = num_cls

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def evaluate(self, batch, stage=None):
        y, emb = batch
        # acuraccy
        logits = self(emb)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, "multiclass", num_classes=self.num_cls)

        if stage:
            self.log(f"{stage}_loss", loss)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            setattr(self, f"{stage}_loss", loss)
            setattr(self, f"{stage}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.evaluate(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.trainer.lr,  # type: ignore
            momentum=0.9,
            weight_decay=self.cfg.trainer.decay,
        )
        scheduler_dict = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.trainer.epoch,  # type: ignore
        )
        return [optimizer], [scheduler_dict]


print("IMPORT FIN")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    cfg = cast(Config, cfg)

    datamodule = DataModule(cfg)
    datamodule.setup()
    model = LinearProbing(cfg, datamodule.input_dim, datamodule.num_classes)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.epoch,
        accelerator="auto",
        # devices=1 if torch.cuda.is_available() else None,  # type: ignore
        callbacks=[
            LearningRateMonitor(logging_interval="step"),  # type: ignore
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        # profiler="simple",
    )

    trainer.fit(model, datamodule)  # type: ignore
    trainer.test(model, datamodule)  # type: ignore

    if hasattr(model, "train_acc") and hasattr(model, "test_acc"):
        assert isinstance(model.val_acc, Tensor) and isinstance(model.test_acc, Tensor)
        val_acc = model.val_acc.item()
        test_acc = model.test_acc.item()
        print([str(val_acc), str(test_acc)])
        with open("results.csv", "r+", newline="") as f:
            reader = csv.reader(f)
            results = list(reader)
            results.append([cfg.version, str(val_acc), str(test_acc)])

            f.seek(0)
            writer = csv.writer(f)
            writer.writerows(results)
            f.truncate()


if __name__ == "__main__":
    main()
