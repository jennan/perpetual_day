from pathlib import Path

import pvlib
import numpy as np
from joblib import Parallel, delayed

import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe

import site_archive_nci  # noqa


def filter_day_time(iterator, bbox):
    """
    Get the valid time where all pixels in the region have SZA <= 60°.

    Parameters:
        iterator (pyearthtools.pipeline.iterators.Iterator): dates to filter
        bbox (tuple): bounding box as (lat_min, lat_max, lon_min, lon_max)

    Returns:
        pyearthtools.pipeline.iterators.Predefined: iterator over valid times
    """
    lat_min, lat_max, lon_min, lon_max = bbox

    bounding_points = [
        (lat_max, lon_min),  # top-left
        (lat_max, lon_max),  # top-right
        (lat_min, lon_min),  # bottom-left
        (lat_min, lon_max),  # bottom-right
    ]
    bounding_points
    lat_array = [point[0] for point in bounding_points]
    lon_array = [point[1] for point in bounding_points]

    valid_times = []
    for i_time in iterator:
        time_array = [i_time.datetime] * len(bounding_points)
        solpos = pvlib.solarposition.get_solarposition(time_array, lat_array, lon_array)
        if (solpos["zenith"] <= 60).all():
            valid_times.append(i_time)

    return petpipe.iterators.Predefined(valid_times)


class XYtoLonLatRectilinear(petdata.transforms.projection.XYtoLonLatRectilinear):
    def __init__(self, projection_method):
        super().__init__(projection_method)
        self.record_initialisation()


class ToFloat32(petpipe.operation.Operation):
    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"name": "FillNan"}}

    def __init__(self):
        super().__init__(
            operation="apply",
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=np.ndarray,
        )
        self.record_initialisation()

    def apply_func(self, sample: np.ndarray):
        return sample.astype(np.float32)


def himawari_pipeline(input_bands, bbox, cachedir):
    satproj = petdata.transforms.projection.HimawariProjAus()
    himawari = petdata.archive.HimawariChannels(bands=input_bands)
    himawari._import = "site_archive_nci"
    pipeline = petpipe.Pipeline(
        himawari,
        XYtoLonLatRectilinear(satproj),
        petdata.transform.region.Bounding(*bbox),
        petdata.transforms.variables.Drop(["x", "y", "geostationary"]),
        petpipe.operations.xarray.conversion.ToNumpy(),
        ToFloat32(),
        petpipe.operations.numpy.reshape.Squeeze(axis=1),
        petpipe.modifications.Cache(
            cachedir, pattern_kwargs={"extension": "npy"}, cache_validity="trust"
        ),
    )
    return pipeline


def features_pipeline(bbox, cachedir):
    """Himawari pipeline of the infrared bands"""
    input_bands = [
        "OBS_B08",
        "OBS_B09",
        "OBS_B10",
        "OBS_B11",
        "OBS_B12",
        "OBS_B13",
        "OBS_B14",
        "OBS_B15",
        "OBS_B16",
    ]
    return himawari_pipeline(input_bands, bbox, cachedir)


def target_pipeline(bbox, cachedir):
    """Himawari pipeline of the visible bands"""
    target_bands = [
        "OBS_B03",
    ]
    return himawari_pipeline(target_bands, bbox, cachedir)


def normed_pipeline(pipeline, cachedir, indices):
    mean_file = cachedir / "mean.npy"
    std_file = cachedir / "std.npy"

    if not mean_file.is_file() or not std_file.is_file():
        # TODO randomize sampling?
        samples = [pipeline[i] for i in indices]
        samples = np.stack(samples)
        mean = samples.mean(axis=(0, 2, 3))[..., None, None]
        std = samples.std(axis=(0, 2, 3))[..., None, None]
        np.save(mean_file, mean)
        np.save(std_file, std)

    pipeline = petpipe.Pipeline(
        pipeline,
        petpipe.operations.numpy.normalisation.Deviation(
            mean_file, std_file, expand=False
        ),
        iterator=pipeline.iterator,
    )

    return pipeline


def full_pipeline(date_range, bbox, cachedir, n_samples=50):
    valid_range = filter_day_time(date_range, bbox)

    cachedir = Path(cachedir)
    featdir = cachedir / "features"
    targetdir = cachedir / "targets"

    featpipe = features_pipeline(bbox, featdir)
    targetpipe = target_pipeline(bbox, targetdir)

    if n_samples is not None:
        indices = list(valid_range)[:n_samples]  # TODO use random dates?
        featpipe = normed_pipeline(featpipe, featdir, indices)
        targetpipe = normed_pipeline(targetpipe, targetdir, indices)

    fullpipe = petpipe.Pipeline((featpipe, targetpipe), iterator=valid_range)

    return fullpipe


def filter_dates(pipeline, n_jobs):
    def good_date(date):
        try:
            pipeline[date]
            return date
        except petdata.exceptions.DataNotFoundError:
            return None

    good_dates = Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(good_date)(date) for date in pipeline.iterator
    )
    good_date = [date for date in good_dates if date is not None]

    return good_date
