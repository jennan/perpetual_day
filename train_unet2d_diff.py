import lightning as L
import pyearthtools.pipeline as petpipe
import pyearthtools.training as pettrain
from sklearn.model_selection import train_test_split
from diffusers import UNet2DModel

from pipeline import full_pipeline, filter_dates
from models import DiffusionModel

if __name__ == "__main__":
    date_range = petpipe.iterators.DateRange(
        "20200101T0000", "20220101T0000", interval="30 minutes"
    )
    bbox = [-35.27, -34, 150, 151.27]
    cachedir = "/scratch/nf33/mr3857/cache2"

    fullpipe = full_pipeline(date_range, bbox, cachedir)
    good_dates = filter_dates(fullpipe, cachedir, n_jobs=12)
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

    unet = UNet2DModel(
        sample_size=features.shape[1],
        in_channels=features.shape[0] + targets.shape[0],
        out_channels=targets.shape[0],
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    model = DiffusionModel(unet, learning_rate=1e-4)

    trainer = L.Trainer(
        max_epochs=1,
        precision="16-mixed",
        callbacks=[L.pytorch.callbacks.ModelCheckpoint()],
        devices=2,
        strategy="ddp",
    )
    trainer.fit(model, dm)

    print("training finished")
