import os
from functools import partial
from os import cpu_count
from typing import cast

import lightning as L
import torch
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms as T
from transformers import AutoProcessor, CLIPModel, ViTImageProcessor, ViTModel

from .config import Config

DATASETS_PATH = "/app/.datasets"
MAX_IMAGES_PER_CLASS = 100  # Adjust as needed


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.ds_dict = get_dataset(self.cfg)
        self.input_dim = torch.tensor(self.ds_dict["train"][0]["embed"]).shape[0]
        self.num_classes = self.ds_dict["train"].features["label"].num_classes
        self.train_ds = MyDataset(self.ds_dict["train"])
        self.val_ds = MyDataset(self.ds_dict["test"], test=True)
        self.test_ds = MyDataset(self.ds_dict["test"], test=True)

    def train_dataloader(self):
        return make_dataloader(self.train_ds, self.cfg, shuffle=True)

    def val_dataloader(self):
        return make_dataloader(self.val_ds, self.cfg, shuffle=False)

    def test_dataloader(self):
        return make_dataloader(self.test_ds, self.cfg, shuffle=False)


class MyDataset(TorchDataset):
    def __init__(self, dataset: Dataset, test=False):
        self.dataset = dataset
        self.trans = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label_tensor = torch.tensor(item["label"])
        embed_tensor = torch.tensor(item["embed"])
        return label_tensor, embed_tensor


def get_dataset(cfg: Config):
    ds_dict = load_raw_dataset(cfg)
    new_ds_dict = DatasetDict()
    for key, dataset in ds_dict.items():
        new_ds_dict[key] = dataset.map(
            get_func(cfg.base_model),
            batched=True,
            batch_size=16,
            with_rank=True,
            num_proc=torch.cuda.device_count(),  # torch.cuda.device_count() # , load_from_cache_file=False
        )
    return new_ds_dict


def get_n_item_per_class(dataset: Dataset, n: int):
    item_dict = {}  # type: ignore
    for item in dataset:
        label = cast(str, item["label"])  # type: ignore
        if label not in item_dict:
            item_dict[label] = []
        if len(item_dict[label]) < n:
            item_dict[label].append(item)
    datasets_list = []

    for class_label, items_list in item_dict.items():
        datasets_list.append(
            Dataset.from_list(items_list, features=dataset.features.copy())
        )
    new_dataset: Dataset = concatenate_datasets(datasets_list)
    new_dataset = new_dataset.shuffle()
    return new_dataset


# ! BASE
# region
def load_raw_dataset(cfg: Config):
    name = cfg.dataloader.name
    if name.split(os.path.sep)[0] == "local":
        ds_dict = cast(
            DatasetDict, load_dataset("imagefolder", data_dir=f"{DATASETS_PATH}/{name}")
        )
    else:
        ds_dict = cast(DatasetDict, load_dataset(name))
    image_key = get_image_key(ds_dict["train"])
    assert image_key is not None, "Image Key not found"
    if image_key != "image":
        for key, dataset in ds_dict.items():
            ds_dict[key] = dataset.rename_column(image_key, "image")
    # return ds_dict

    if cfg.dataloader.shot is not None:
        limited_ds_dict = DatasetDict()
        for key, dataset in ds_dict.items():
            assert isinstance(dataset, Dataset)
            limited_ds = dataset
            if key == "train":
                limited_ds = get_n_item_per_class(dataset, cfg.dataloader.shot)
            limited_ds_dict[key] = limited_ds
        return limited_ds_dict
    else:
        return ds_dict


def embed(batch, rank: int, model: ViTModel, processor: ViTImageProcessor):
    device = f"cuda:{(rank or 0)}"
    model.to(device)  # type: ignore

    def v_embed(x):
        inputs = processor(images=x, return_tensors="pt").to(device)
        emb = model(**inputs).last_hidden_state[:, 0][0]
        return emb

    batch["embed"] = [v_embed(x) for x in batch["image"]]
    return batch


def get_func(name: str) -> partial:
    processor = ViTImageProcessor.from_pretrained(name)
    model = ViTModel.from_pretrained(name)
    return partial(embed, model=model, processor=processor)  # type: ignore


WORKER_DIV = 1
DRY = False


def make_dataloader(dataset: TorchDataset, cfg: Config, shuffle: bool = True):
    cpu_cnt = cpu_count()
    assert cpu_cnt is not None

    return DataLoader(
        dataset,  # type: ignore
        batch_size=cfg.dataloader.batch_size if shuffle else 1024,
        num_workers=1 if DRY else cpu_cnt // (2 * WORKER_DIV),
        prefetch_factor=1 if DRY else cpu_cnt // (1 * WORKER_DIV),
        persistent_workers=False if DRY else True,
        pin_memory=False if DRY else True,
        shuffle=shuffle,
    )


# endregion


# ! UTILS
# region
def get_image_key(ds: Dataset) -> str:
    for key in ["image", "img", "Images"]:
        if key in ds.features.keys():
            return key
    assert False, "No image key found"  # type: ignore


# endregion
