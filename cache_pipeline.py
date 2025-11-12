import dask
from dask.distributed import Client

import pyearthtools.pipeline as petpipe

from pipeline import full_pipeline


if __name__ == "__main__":
    date_range = petpipe.iterators.DateRange(
        "20200101T0000", "20200201T0000", interval="30 minutes"
    )
    bbox = [-35.27, -34, 150, 151.27]
    cachedir = "/scratch/nf33/mr3857/cache"

    fullpipe = full_pipeline(date_range, bbox, cachedir, clean_cache=False)

    def cache_data(date):
        with dask.config.set(scheduler="single-threaded"):
            fullpipe[date]

    client = Client(n_workers=6, threads_per_worker=2)
    print(client.dashboard_link)

    futures = client.map(cache_data, list(fullpipe.iterator))
    client.gather(futures)
