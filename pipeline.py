import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe

import site_archive_nci  # noqa


def filter_days(iterator, bbox):
    """
    Get the valid time where all pixels in the region have SZA <= 60°.
    
    Parameters:
        region (tuple): (lat_min, lat_max, lon_min, lon_max).
        start_time (str).
        end_time (str).
    
    Returns:
        list: Valid times.
    """
    lat_min, lat_max, lon_min, lon_max = region
    
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
    for i_time in date_range:
        time_array = [i_time.datetime] * len(bounding_points)
        solpos = pvlib.solarposition.get_solarposition(time_array, lat_array, lon_array)
        print(solpos['zenith'])
        if (solpos['zenith'] <= 60).all():
            valid_times.append(i_time)
    return iterator


def features_pipeline(date_range, bbox):
    """Himawari pipeline of the infrared bands"""
    # TODO add normalisation

    # filter days
    valid_dates = filter_days(date_range, bbox)

    # create pipeline
    satproj = petdata.transforms.projection.HimawariProjAus()
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

    pipeline = petpipe.Pipeline(
        petdata.archive.HimawariChannels(bands=input_bands),
        petdata.transforms.projection.XYtoLonLatRectilinear(satproj),
        petdata.transform.region.Bounding(*bbox),
        petdata.transforms.variables.Drop(["x", "y", "geostationary"]),
        petpipe.operations.xarray.conversion.ToNumpy(),
        petpipe.operations.numpy.reshape.Squeeze(axis=1),
        iterator=valid_dates,
    )

    return pipeline


def target_pipeline(iterator, bbox):
    # create pipeline
    satproj = petdata.transforms.projection.HimawariProjAus()
    target_bands = [
        "OBS_B03",
    ]

    pipeline = petpipe.Pipeline(
        petdata.archive.HimawariChannels(bands=target_bands),
        petdata.transforms.projection.XYtoLonLatRectilinear(satproj),
        petdata.transform.region.Bounding(*bbox),
        petdata.transforms.variables.Drop(["x", "y", "geostationary"]),
        petpipe.operations.xarray.conversion.ToNumpy(),
        petpipe.operations.numpy.reshape.Squeeze(axis=1),
    )

    return pipeline
