

def filter_days(iterator, bbox):
    # TODO implement filtering
	return iterator


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

