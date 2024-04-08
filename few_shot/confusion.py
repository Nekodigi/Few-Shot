import numpy as np
import seaborn as sns
import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from few_shot.config import Config
from few_shot.dataset import DataModule


def class_wise_acc(cfg, model, datamodule: DataModule, device):
    train_db = datamodule.ds_dict["train"]
    train_db.add_faiss_index("embed")
    class_acc_list, y_preds, true_label = [], [], []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for imgs, labels, embed in datamodule.test_dataloader():
            embed = embed.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            if cfg.training_type == "rag":
                scores, retrieved = train_db.get_nearest_examples_batch(
                    "embed", embed.cpu().numpy(), 1
                )
                preds = torch.cat([torch.tensor(data["label"]) for data in retrieved])
            else:
                preds = torch.argmax(logits, dim=1)
            y_preds.extend(preds.cpu().numpy())
            true_label.extend(labels.cpu().numpy())
            # print(preds[:4], labels[:4])
        cf = confusion_matrix(true_label, y_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = np.divide(
            cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt != 0
        )
        # cls_acc = cls_hit / cls_cnt
        class_acc_list.append(cls_acc)
    model.train()
    return (
        class_acc_list[0],
        y_preds,
        true_label,
        np.round(confusion_matrix(true_label, y_preds), 2),
    )


def save_confusion_matrix(
    cfg: Config, model: LightningModule, datamodule: DataModule, file_name: str
):
    acc, pred, label, cm = class_wise_acc(cfg, model, datamodule, "cuda:0")
    # visualize cm
    plt.figure(figsize=(10, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=range(datamodule.num_classes),  # type: ignore
        yticklabels=range(datamodule.num_classes),  # type: ignore
    )
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    # save as img
    plt.savefig(file_name)
    plt.show()
