import torch
from torch.utils.tensorboard import SummaryWriter

from few_shot.config import Config
from few_shot.dataset import DataModule, apply_image_rect_trans, get_image_key


class Projector:
    def __init__(self, cfg: Config, datamodule: DataModule, save_dir: str) -> None:
        self.cfg = cfg
        self.datamodule = datamodule
        self.save_dir = save_dir

    def __enter__(self):
        self.writer = SummaryWriter(
            f"{self.save_dir}/{self.cfg.dataloader.name.replace('/', '_')}"
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.flush()
        self.writer.close()
        print("==Done==")

    def project_random_n(self, n) -> None:
        dataset = self.datamodule.ds_dict["train"]
        image_key = get_image_key(dataset)  # type: ignore
        apply_image_rect_trans(dataset, image_key, 128)

        selected = dataset.shuffle(100).select(range(n))  # type: ignore

        images: torch.Tensor = torch.stack(selected[:]["image"])
        label = selected[:]["label"]
        embed = torch.tensor(selected[:]["embed"])

        print("==Writing==")
        self.writer.add_embedding(
            embed,
            metadata=label,
            label_img=images,
            global_step=f"{self.cfg.base_model.replace('/', '_')}",  # f"{self.cfg.dataloader.name.replace('/', '_')}",
            # tag=,
        )
