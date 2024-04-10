import csv

import lightning as L
from torch import Tensor

from few_shot.config import Config


def log_model_acc(cfg: Config, model: L.LightningModule, minimal=False):
    val_acc = 0
    test_acc = 0
    if hasattr(model, "train_acc") and isinstance(model.val_acc, Tensor):
        val_acc = model.val_acc.item()
    if hasattr(model, "test_acc") and isinstance(model.test_acc, Tensor):
        test_acc = model.test_acc.item()
    print([str(val_acc), str(test_acc)])
    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        v = cfg.version
        if minimal:
            v = v.split("/")[-1]
        writer.writerow([v, str(val_acc), str(test_acc)])


def log_config(cfg: Config):
    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([cfg.version])
