import sys

import yaml
import pyearthtools.pipeline as petpipe
from sklearn.model_selection import train_test_split

from config import Config
from pipeline import full_pipeline, filter_dates
from plot import plot_results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Missing configuration file!", file=sys.stderr)
        print(f"Usage: {sys.argv[0]} <config_file.yaml>", file=sys.stderr)
        sys.exit(1)

    print("predictions started")

    with open(sys.argv[1], "r") as fd:
        config = Config(**yaml.safe_load(fd))

    modelpath = config.resultsdir / "lightning_logs/version_0/checkpoints/last.ckpt"
    model = config.model.load_from_checkpoint(modelpath)

    date_range = petpipe.iterators.DateRange(
        config.start_date, config.end_date, config.interval
    )
    fullpipe = full_pipeline(date_range, config.bbox, config.cachedir)
    good_dates = filter_dates(fullpipe, config.cachedir, n_jobs=config.n_jobs)
    train_dates, valid_dates = train_test_split(
        good_dates, random_state=config.seed, shuffle=True, test_size=config.test_size
    )

    # get the feature pipeline (without cache) to undo transforms on features
    featurepipe = fullpipe.steps[0].sub_pipelines[0]
    featurepipe_nocache = petpipe.Pipeline(
        *featurepipe.steps[:-2], featurepipe.steps[-1]
    )
    featurepipe_nocache[good_dates[0]]

    # get the target pipeline (without cache) to undo transforms on predictions
    targetpipe = fullpipe.steps[0].sub_pipelines[1]
    targetpipe_nocache = petpipe.Pipeline(*targetpipe.steps[:-2], targetpipe.steps[-1])
    targetpipe_nocache[good_dates[0]]

    for date in train_dates[:5]:
        features, targets = fullpipe[date]
        preds = model.predict(features, targets.shape)

        features = featurepipe_nocache.undo(features).isel(time=0)
        targets = targetpipe_nocache.undo(targets).isel(time=0)
        preds = targetpipe_nocache.undo(preds).isel(time=0)

        fig = plot_results(features, targets, preds)
        figname = config.resultsdir / f"train_{date.datetime:%Y%m%dT%H%M}.png"
        fig.savefig(figname, bbox_inches="tight")

    for date in valid_dates[:5]:
        features, targets = fullpipe[date]
        preds = model.predict(features, targets.shape)

        features = featurepipe_nocache.undo(features).isel(time=0)
        targets = targetpipe_nocache.undo(targets).isel(time=0)
        preds = targetpipe_nocache.undo(preds).isel(time=0)

        fig = plot_results(features, targets, preds)
        figname = config.resultsdir / f"val_{date.datetime:%Y%m%dT%H%M}.png"
        fig.savefig(figname, bbox_inches="tight")

    print("predictions finished")
