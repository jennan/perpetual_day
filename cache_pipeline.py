from pathlib import Path

import dask
from dask.distributed import Client

import pyearthtools.pipeline as petpipe
import pyearthtools.training as pettrain
from sklearn.model_selection import train_test_split

from pipeline import full_pipeline


if __name__ == "__main__":
    date_range = petpipe.iterators.DateRange('20200101T0000', '20200201T0000', interval='30 minutes')
    bbox = [-35.27, -34, 150, 151.27]
    cachedir = "/scratch/nf33/mr3857/cache"

    fullpipe = full_pipeline(date_range, bbox, cachedir, clean_cache=False)

    # train_dates, valid_dates = train_test_split(
    #     fullpipe.iterator.samples, random_state=42, shuffle=True, test_size=0.2
    # )
    # train_split = petpipe.iterators.Predefined(train_dates)
    # valid_split = petpipe.iterators.Predefined(valid_dates)

    # dm = pettrain.data.lightning.PipelineLightningDataModule(
    #     fullpipe,
    #     train_split=train_split,
    #     valid_split=valid_split,
    #     batch_size=4,
    #     num_workers=8,
    #     shuffle=True,
    #     multiprocessing_context="forkserver",
    #     persistent_workers=True,
    # )

    # dataloader = dm.train_dataloader()

    # for batch_idx, (inputs, targets) in enumerate(dataloader):
    #     print(batch_idx, inputs.shape, targets.shape)
    #     break

    def cache_data(date):
        with dask.config.set(scheduler="single-threaded"): 
            fullpipe[date]
 
    client = Client(n_workers=6, threads_per_worker=2)
    print(client.dashboard_link)

    futures = client.map(cache_data, list(fullpipe.iterator))
    client.gather(futures)
