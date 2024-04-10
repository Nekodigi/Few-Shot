from sys import settrace
from typing import cast

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging

from few_shot.config import Config
from few_shot.confusion import log_confusion_matrix
from few_shot.dataset import DataModule
from few_shot.linear_probing import LinearProbing
from few_shot.make_dataset import DatasetMaker
from few_shot.projector import Projector
from few_shot.rag import RAG
from few_shot.utils import log_config, log_model_acc

settrace

print("IMPORT FIN")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    cfg = cast(Config, cfg)

    dataset_maker = DatasetMaker()

    log_config(cfg)

    for i in range(5):
        dataset_name = dataset_maker.classify("size", "single", i).make_dataset()
        cfg.dataloader.name = dataset_name
        print(f"====Training {cfg.dataloader.name}====")
        train(cfg)


def train(cfg: Config):
    datamodule = DataModule(cfg)
    datamodule.setup()

    # visualize embeddings
    with Projector(
        cfg, datamodule, "projector"  # /{cfg.dataloader.name}/{cfg.base_model}
    ) as projector:
        projector.project_random_n(25)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.epoch,
        accelerator="auto",
        callbacks=[
            LearningRateMonitor(logging_interval="step"),  # type: ignore
            StochasticWeightAveraging(swa_lrs=cfg.trainer.swa),
        ],
        log_every_n_steps=1,
        # profiler="simple",
    )

    if cfg.training_type == "rag":
        train_db = datamodule.ds_dict["train"]
        train_db.add_faiss_index("embed")
        model = RAG(cfg, datamodule.num_classes, train_db)
    else:
        model = LinearProbing(cfg, datamodule.input_dim, datamodule.num_classes)
        trainer.fit(model, datamodule)  # type: ignore

    trainer.test(model, datamodule)  # type: ignore

    log_model_acc(cfg, model, True)
    log_confusion_matrix(cfg, model, datamodule)


if __name__ == "__main__":
    main()
