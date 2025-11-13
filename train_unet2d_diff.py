from pathlib import Path

import torch
import lightning as L
import pyearthtools.pipeline as petpipe
import pyearthtools.training as pettrain
from tqdm.notebook import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from diffusers import UNet2DModel

from src.pipeline import full_pipeline, filter_dates
from src.models import DiffusionModel
from src.plot import plot_results


def predict(model, features, targets):
    model = model.cuda()
    x = torch.randn(*targets.shape).unsqueeze(0).cuda()
    y = torch.from_numpy(features).unsqueeze(0).cuda()
    for i, t in tqdm(enumerate(model.scheduler.timesteps)):
        with torch.no_grad():
            residual = model(x, t, y)
        x = model.scheduler.step(residual, t, x).prev_sample
    preds = x.squeeze(0).cpu().detach().numpy()
    return preds


if __name__ == "__main__":
    date_range = petpipe.iterators.DateRange(
        "20200101T0000", "20220101T0000", interval="30 minutes"
    )
    bbox = [-35.27, -34, 150, 151.27]
    cachedir = "/scratch/nf33/mr3857/cache2"

    exp_id = 1
    while (resultsdir := Path("results") / f"unet2_diff_{exp_id:03}").is_dir():
        exp_id += 1
    resultsdir.mkdir(exist_ok=True, parents=True)
    print("results dir:", resultsdir)

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

    checkpoint = ModelCheckpoint(save_top_k=1, save_last=True, monitor="val_loss")
    trainer = L.Trainer(
        max_epochs=1,
        precision="16-mixed",
        callbacks=[checkpoint],
        # devices=2,
        # strategy="ddp",
        default_root_dir=resultsdir,
    )
    trainer.fit(model, dm)

    print("training finished")

    # get the target pipeline (without cache) to undo transforms on predictions
    targetpipe = fullpipe.steps[0].sub_pipelines[1]
    targetpipe = petpipe.Pipeline(*targetpipe.steps[:-2], targetpipe.steps[-1])
    targetpipe["20200301T0000"]

    # get the feature pipeline (without cache) to undo transforms on features
    featurepipe = fullpipe.steps[0].sub_pipelines[0]
    featurepipe = petpipe.Pipeline(*featurepipe.steps[:-2], featurepipe.steps[-1])
    featurepipe["20200301T0000"]

    dm.train()
    for i in range(10):
        features, targets = dm[i]
        preds = predict(model, features, targets)

        features = featurepipe.undo(features).isel(time=0)
        targets = targetpipe.undo(targets).isel(time=0)
        preds = targetpipe.undo(preds).isel(time=0)

        fig = plot_results(features, targets, preds)
        fig.savefig(resultsdir / f"train_{i}.png", bbox_inches="tight")

    dm.eval()
    for i in range(10):
        features, targets = dm[i]
        preds = predict(model, features, targets)

        features = featurepipe.undo(features).isel(time=0)
        targets = targetpipe.undo(targets).isel(time=0)
        preds = targetpipe.undo(preds).isel(time=0)

        fig = plot_results(features, targets, preds)
        fig.savefig(resultsdir / f"val_{i}.png", bbox_inches="tight")
