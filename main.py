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
from few_shot.confusion import save_confusion_matrix
from few_shot.dataset import DataModule
from few_shot.projector import Projector

settrace


class LinearProbing(L.LightningModule):
    def __init__(self, cfg: Config):  # , datamodule: DataModule
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(datamodule.input_dim, datamodule.num_classes),
        )
        self.cfg = cfg
        self.lr = cfg.trainer.lr
        self.num_cls = datamodule.num_classes

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def evaluate(self, batch, stage=None):
        _, y, emb = batch
        # acuraccy
        logits = self(emb)
        loss = F.nll_loss(logits, y)
        if self.cfg.training_type == "rag":
            scores, retrieved = train_db.get_nearest_examples_batch(
                "embed", emb.cpu().numpy(), 1
            )
            preds = torch.cat([torch.tensor(data["label"]) for data in retrieved])
        else:
            preds = torch.argmax(logits, dim=1)

        acc = accuracy(preds.cpu(), y.cpu(), "multiclass", num_classes=self.num_cls)

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

    global datamodule
    datamodule = DataModule(cfg)
    datamodule.setup()

    with Projector(
        cfg, datamodule, "projector"  # /{cfg.dataloader.name}/{cfg.base_model}
    ) as projector:
        projector.project_random_n(50)

    global train_db
    train_db = datamodule.ds_dict["train"]
    train_db.add_faiss_index("embed")

    model = LinearProbing(cfg)

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

    if cfg.training_type != "rag":
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

    save_confusion_matrix(cfg, model, datamodule, "confusion_matrix.png")


if __name__ == "__main__":
    main()
