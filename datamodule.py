from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

import pyearthtools.data as petdata


def cache_pipeline(pipeline, cache_dir, n_jobs):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_data(date):
        date_str = str(date).replace(":", "")
        fname_feats = cache_dir / f"features_{date_str}.npy"
        fname_targets = cache_dir / f"targets_{date_str}.npy"

        if fname_feats.is_file() and fname_targets.is_file():
            return

        try:
            features, targets = pipeline[date]
            if not np.isnan(features).any() and not np.isnan(targets).any():
                np.save(fname_feats, features)
                np.save(fname_targets, targets)
        except petdata.exceptions.DataNotFoundError:
            pass

    Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(cache_data)(date) for date in pipeline.iterator
    )


class NpyDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = [
            torch.from_numpy(np.load(fname)).float() for fname in self.file_list[idx]
        ]
        return sample


class NpyDataModule(L.LightningDataModule):
    def __init__(
        self, pipeline, file_dir, val_split: 0.1, test_split: 0.2, **kwargs
    ):
        super().__init__()
        self.pipeline = pipeline
        self.file_dir = Path(file_dir)
        self.val_split = val_split
        self.test_split = test_split
        self.kwargs = kwargs
        self.file_list = []

    def prepare_data(self):
        features_list = sorted(self.file_dir.glob("features_*.npy"))
        targets_list = sorted(self.file_dir.glob("targets_*.npy"))
        self.file_list = list(zip(features_list, targets_list))

    def setup(self, stage: str):
        train_files, test_files = train_test_split(
            self.file_list, random_state=42, shuffle=True, test_size=self.test_split
        )
        train_files, val_files = train_test_split(
            train_files, random_state=42, shuffle=True, test_size=self.val_split
        )
        self.test_ds = NpyDataset(test_files)
        self.val_ds = NpyDataset(val_files)
        self.train_ds = NpyDataset(train_files)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.kwargs)
