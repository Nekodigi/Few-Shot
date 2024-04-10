import lightning as L
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import Tensor, nn
from torchmetrics.functional import accuracy

from few_shot.config import Config


class LinearProbing(L.LightningModule):
    def __init__(
        self, cfg: Config, input_dim: int, num_classes: int
    ):  # , datamodule: DataModule
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            # nn.Dropout(cfg.trainer.dropout),
        )
        self.cfg = cfg
        self.lr = cfg.trainer.lr
        self.num_cls = num_classes
        model = timm.create_model(cfg.base_model, pretrained=True)
        model.eval()
        self.trans = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        self.base_model = model.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.stack([self.trans(T.ToPILImage()(img)) for img in x]).to(self.device)  # type: ignore
        emb = self.base_model.forward_features(x)
        emb = emb[:, 0]

        out = self.model(emb)
        return F.log_softmax(out, dim=1)

    def evaluate(self, batch, stage=None):
        imgs, y, emb = batch
        # to pil

        # acuraccy
        logits = self(imgs)
        loss = F.nll_loss(logits, y)
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
