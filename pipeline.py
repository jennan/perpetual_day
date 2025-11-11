

def filter_days(iterator, bbox):
	pass


def features_pipeline(date_range, bbox):

	# filter days
    valid_dates = filter_days(date_range, bbox)

	# create pipeline
    pipeline = Pipeline(
        # TODO steps
        iterator=valid_dates,
    )

	return pipeline


def target_pipeline(iterator, bbox):
	pass

