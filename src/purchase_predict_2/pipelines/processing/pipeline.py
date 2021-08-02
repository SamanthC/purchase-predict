from kedro.pipeline import Pipeline, node

from .nodes import encode_feature, split_dataset

def create_pipeline(*kwargs):
	return Pipeline(
		[
			node(
				encode_feature,
				"primary",
				dict(features="dataset", transform_pipeline="transform_pipeline")	
			),
			node(
				split_dataset,
				["dataset", "params:test_ratio"],
				dict(
					X_train="X_train",
					y_train="y_train",
					X_test="X_test",
					y_test="y_test"
				)

			)
		]
	)
