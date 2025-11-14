import pickle
from pathlib import Path

import lightning as L
import pyearthtools.pipeline as petpipe
import pyearthtools.training as pettrain
from sklearn.model_selection import train_test_split
from diffusers import UVit2DModel

from pipeline import full_pipeline, filter_dates
from models import DiffusionUVit2DModel

if __name__ == "__main__":
    date_range = petpipe.iterators.DateRange(
        "20200101T0000", "20220101T0000", interval="30 minutes"
    )
    bbox = [-35.27, -34, 150, 151.27]
    cachedir = "/scratch/nf33/yl4436/cache2"

    fullpipe = full_pipeline(date_range, bbox, cachedir)

    # TODO use dates in filename
    good_dates_file = Path("valid_dates.pkl")

    if not good_dates_file.is_file():
        good_dates = filter_dates(fullpipe, n_jobs=12)
        with good_dates_file.open("wb") as fd:
            pickle.dump(good_dates, fd)

    with good_dates_file.open("rb") as fd:
        good_dates = pickle.load(fd)

    print("good dates:", len(good_dates))

    train_dates, valid_dates = train_test_split(
        good_dates, random_state=42, shuffle=True, test_size=0.1
    )
    train_split = petpipe.iterators.Predefined(train_dates)
    valid_split = petpipe.iterators.Predefined(valid_dates)

    dm = pettrain.data.lightning.PipelineLightningDataModule(
        fullpipe,
        train_split=train_split,
        valid_split=valid_split,
        batch_size=64,
        num_workers=6,
        shuffle=True,
        multiprocessing_context="forkserver",
        persistent_workers=True,
    )

    features, targets = next(iter(fullpipe))
    print(f"features shape and type: {features.shape}, {features.dtype}")
    print(f"targets shape and type: {targets.shape}, {targets.dtype}")
    
    uvit = UVit2DModel(
        sample_size=features.shape[1],        # image size
        hidden_size=1024,      # transformer hidden size
        num_hidden_layers=22,  # depth
        num_attention_heads=16 # attention heads
    )

    model = DiffusionUVit2DModel(uvit, learning_rate=1e-4)

    trainer = L.Trainer(
        max_epochs=1,
        precision="16-mixed",
        callbacks=[L.pytorch.callbacks.ModelCheckpoint()],
        # devices=2,
        # strategy="ddp",
    )
    trainer.fit(model, dm)

    print("training finished")
