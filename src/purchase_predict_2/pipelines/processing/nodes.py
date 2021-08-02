import pandas as pd 
from typing import Any, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_feature(dataset: pd.DataFrame) -> pd.DataFrame:
	"""
	Encode feature of data file
	"""
	features = dataset.drop(["user_id", "user_session"], axis=1).copy()

	encoders = []
	for label in ["category", "sub_category", "brand"]:
		features[label] = features[label].astype(str)
		features.loc[features[label] == "nan", label] = "unknown"
		encoder = LabelEncoder()
		features.loc[:, label] = encoder.fit_transform(features.loc[:, label].copy())
		encoders.append((label, encoder))

	features["weekday"] = features["weekday"].astype(int)
	return dict(features = features, transform_pipeline = encoders)

def split_dataset(dataset: pd.DataFrame, test_ratio: float) -> Dict[str, Any]:
	"""
	Splis dataset into a training set and a test set
	"""

	X = dataset.drop("purchased", axis=1)
	y = dataset["purchased"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state = 40)

	return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)