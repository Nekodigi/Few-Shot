import os
import shutil
from typing import cast

import pandas as pd
from scipy import stats

METHODS = ["single", "avg", "mode"]


class DatasetMaker:
    def __init__(self):
        self.df = pd.read_csv(".datasets/original/split.csv", delimiter="\t")
        columns = self.df.columns
        self.columns = columns[: int(len(columns) / 2 + 1)]

    def get_lut(self, category: str):
        # like ["normal", "dry", "wet", "NA", "NA"]
        raw_lut = self.df[f"{category}_lut"]
        lut = []
        for i in range(5):
            lut.append("NA" if raw_lut.isna()[i] else raw_lut[i])
        return lut

    def get_digits(self, number: int):
        return [int(d) for d in str(number)]

    def get_abbr(self, category: str, method: str, arg: int = 0):
        if method == "single":
            return f"{category}/{method}{arg}"
        else:
            return f"{category}/{method}"

    def classify(
        self, category: str, method: str, arg: int = 0, apply_lut: bool = True
    ):
        data = self.df[category]
        results = []
        for d in data:
            value = 5
            digits = self.get_digits(cast(int, d))
            if method == "single":
                value = digits[arg]
            elif method == "avg":
                value = round(sum(digits) / len(digits))
            elif method == "mode":
                value = stats.mode(digits).mode

            results.append(value)
        if apply_lut:
            results = self.apply_lut(results, category)
        return DatasetOrigin(results, self.get_abbr(category, method, arg))

    def apply_lut(self, lst: list[int], category: str):
        return [self.get_lut(category)[i - 1] for i in lst]


class DatasetOrigin:
    def __init__(self, lst: list[str], name: str):
        self.lst = lst
        self.train: dict[str, list[int]] = {}
        self.test: dict[str, list[int]] = {}
        self.name = name
        self.image_dir = ".datasets/original/images"
        self.dataset_name = f"local/tongue/{name}"
        self.dataset_dir = f".datasets/{self.dataset_name}"

    def __str__(self) -> str:
        return f"{self.name}\n{str(self.lst)}"

    def test_split(self, n_test: int = 5):
        # spare each n_item for test
        for i, item in enumerate(self.lst):
            # make sure test has [item] key
            if item not in self.test:
                self.test[item] = [i]
            elif len(self.test[item]) < n_test:
                self.test[item] += [i]
            else:
                if item not in self.train:
                    self.train[item] = []
                self.train[item] += [i]

    # TODO: Apply mask??
    def make_dataset(self, n_test: int = 5):
        self.test_split(n_test)
        self.split = {"train": self.train, "test": self.test}
        for s in ["train", "test"]:
            for key, lst in self.split[s].items():
                os.makedirs(f"{self.dataset_dir}/{s}/{key}", exist_ok=True)
                for item in lst:
                    shutil.copy(
                        f"{self.image_dir}/{item}.png",
                        f"{self.dataset_dir}/{s}/{key}/{item}.png",
                    )
        return self.dataset_name


# dataset_maker = DatasetMaker()

# print(dataset_maker.get_lut("size"))
# print(dataset_maker.classify("size", "single", 1).make_dataset())  #
