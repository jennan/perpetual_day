from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

import pyearthtools.data as petdata


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


class PetDataModule(L.LightningDataModule):
    def __init__(
        self, pipeline, cache_dir, n_jobs, val_split: 0.1, test_split: 0.2, **kwargs
    ):
        super().__init__()
        self.pipeline = pipeline
        self.cache_dir = Path(cache_dir)
        self.n_jobs = n_jobs
        self.val_split = val_split
        self.test_split = test_split
        self.kwargs = kwargs
        self.file_list = []

    def _cache_data(self, date, i):
        try:
            features, targets = self.pipeline[date]
            if np.isnan(features).any():
                filename = None
            elif np.isnan(targets).any():
                filename = None
            else:
                date_str = str(date).replace(":", "")
                fname_feats = self.cache_dir / f"features_{date_str}.npy"
                np.save(fname_feats, features)
                fname_targets = self.cache_dir / f"targets_{date_str}.npy"
                np.save(fname_targets, targets)
                filename = (fname_feats, fname_targets)
        except petdata.exceptions.DataNotFoundError:
            filename = None
        return filename

    def prepare_data(self):
        if self.cache_dir.is_dir():
            features_list = sorted(self.cache_dir.glob("features_*.npy"))
            targets_list = sorted(self.cache_dir.glob("targets_*.npy"))
            self.file_list = list(zip(features_list, targets_list))
        else:
            self.cache_dir.mkdir(parents=True)
            date_range = self.pipeline.iterator
            file_list = Parallel(n_jobs=self.n_jobs, verbose=True)(
                delayed(self._cache_data)(date, i) for i, date in enumerate(date_range)
            )
            self.file_list = sorted([fname for fname in file_list if fname is not None])

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
