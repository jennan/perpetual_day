import pvlib

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
        (lat_min, lon_max)   # bottom-right
    ]
    bounding_points
    lat_array = [point[0] for point in bounding_points]
    lon_array = [point[1] for point in bounding_points]
    
    valid_times = []
    for i_time in iterator:
        time_array = [i_time.datetime] * len(bounding_points)
        solpos = pvlib.solarposition.get_solarposition(time_array, lat_array, lon_array)
        if (solpos['zenith'] <= 60).all():
            valid_times.append(i_time)

    return petpipe.iterators.Predefined(valid_times)


def himawari_pipeline(input_bands, bbox):
    # TODO set MagicNorm cachedir to a specific folder?
    satproj = petdata.transforms.projection.HimawariProjAus()
    pipeline = petpipe.Pipeline(
        petdata.archive.HimawariChannels(bands=input_bands),
        petdata.transforms.projection.XYtoLonLatRectilinear(satproj),
        petdata.transform.region.Bounding(*bbox),
        petdata.transforms.variables.Drop(["x", "y", "geostationary"]),
        petpipe.operations.xarray.normalisation.MagicNorm(),
        petpipe.operations.xarray.conversion.ToNumpy(),
        petpipe.operations.numpy.reshape.Squeeze(axis=1),
    )
    return pipeline


def features_pipeline(bbox):
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
    return himawari_pipeline(input_bands, bbox)


def target_pipeline(bbox):
    """Himawari pipeline of the visible bands"""
    target_bands = [
        "OBS_B03",
    ]
    return himawari_pipeline(target_bands, bbox)
