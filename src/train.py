import sys

import yaml
import lightning as L
import pyearthtools.pipeline as petpipe
import pyearthtools.training as pettrain
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from config import Config
from pipeline import full_pipeline, filter_dates


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Missing configuration file!", file=sys.stderr)
        print(f"Usage: {sys.argv[0]} <config_file.yaml>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r") as fd:
        config = Config(**yaml.safe_load(fd))

    print("full configuration:")
    print(config.model_dump_json(indent=4))

    # create result folder, crash if exists to avoid overriding results
    config.resultsdir.mkdir(exist_ok=False, parents=True)

    date_range = petpipe.iterators.DateRange(
        config.start_date, config.end_date, config.interval
    )
    fullpipe = full_pipeline(date_range, config.bbox, config.cachedir)
    good_dates = filter_dates(fullpipe, config.cachedir, n_jobs=config.n_jobs)
    print("good dates:", len(good_dates))

    train_dates, valid_dates = train_test_split(
        good_dates, random_state=config.seed, shuffle=True, test_size=config.test_size
    )
    train_split = petpipe.iterators.Predefined(train_dates)
    valid_split = petpipe.iterators.Predefined(valid_dates)

    dm = pettrain.data.lightning.PipelineLightningDataModule(
        fullpipe,
        train_split=train_split,
        valid_split=valid_split,
        **{"shuffle": True, **config.datamodule_params},
    )

    features, targets = next(iter(fullpipe))
    print(f"features shape and type: {features.shape}, {features.dtype}")
    print(f"targets shape and type: {targets.shape}, {targets.dtype}")

    model = config.model(**config.model_params)

    checkpoint = ModelCheckpoint(save_top_k=1, save_last=True, monitor="val_loss")
    trainer = L.Trainer(
        callbacks=[checkpoint],
        default_root_dir=config.resultsdir,
        **config.trainer_params,
    )
    trainer.fit(model, dm, ckpt_path=config.ckpt_path)

    print("training finished")
