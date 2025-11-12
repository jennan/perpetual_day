import dask
from dask.distributed import Client

import pyearthtools.pipeline as petpipe
import pyearthtools.data as petdata

from pipeline import full_pipeline


if __name__ == "__main__":
    date_range = petpipe.iterators.DateRange(
        "20200101T0000", "20200201T0000", interval="30 minutes"
    )
    bbox = [-35.27, -34, 150, 151.27]
    cachedir = "/scratch/nf33/mr3857/cache2"

    fullpipe = full_pipeline(
        date_range, bbox, cachedir, clean_cache=False, n_samples=None
    )

    def cache_data(date):
        try:
            with dask.config.set(scheduler="single-threaded"):
                fullpipe[date]
        except petdata.exceptions.DataNotFoundError:
            pass

    with Client(n_workers=6, threads_per_worker=2) as client:
        print(client.dashboard_link)
        futures = client.map(cache_data, list(fullpipe.iterator))
        client.gather(futures)
