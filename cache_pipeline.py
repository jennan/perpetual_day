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
    cachedir = "/scratch/nf33/mr3857/cache"

    fullpipe = full_pipeline(date_range, bbox, cachedir, clean_cache=False)

    def cache_data(date):
        try:
            with dask.config.set(scheduler="single-threaded"):
                fullpipe[date]
                missing_date = None
        except petdata.exceptions.DataNotFoundError:
            missing_date = date
        return missing_date

    client = Client(n_workers=6, threads_per_worker=2)
    print(client.dashboard_link)

    futures = client.map(cache_data, list(fullpipe.iterator))
    results = client.gather(futures)
    # TODO fix printing missing data
    missing = [date for date in results if date is not None]
    print("\n".join([str(date) for date in missing]))
