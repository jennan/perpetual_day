

def filter_days(iterator, bbox):
    # TODO implement filtering
	return iterator


def features_pipeline(date_range, bbox):

	# filter days
    valid_dates = filter_days(date_range, bbox)

	# create pipeline
    satproj = petdata.transforms.projection.HimawariProjAus()
    sat_projector = petdata.transforms.projection.XYtoLonLatRectilinear(satproj)
    # Get the Himawari pipeline of the infrared bands
    input_bands = ['OBS_B08','OBS_B09','OBS_B10','OBS_B11','OBS_B12','OBS_B13','OBS_B14','OBS_B15','OBS_B16']
    sat_accessor_IR = petdata.archive.HimawariChannels(bands=input_bands)
    pipeline = petpipe.Pipeline(
        sat_accessor_IR,
        sat_projector,
        petdata.transform.region.Bounding(-35, -25, 138, 150),  # cut down on region for example
        petdata.transforms.variables.Drop(['x', 'y','geostationary']),
        # petpipe.operations.xarray.normalisation.SingleValueDivision(1200),
        petpipe.operations.xarray.conversion.ToNumpy(),
        petpipe.operations.numpy.reshape.Squeeze(axis=1),
        iterator=valid_dates,
    )
	return pipeline

def target_pipeline(iterator, bbox):
	pass

