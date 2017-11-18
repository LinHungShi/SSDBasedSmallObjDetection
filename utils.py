import pandas as pd
from sklearn.utils import shuffle
import glob
import shutil
import numpy as np

def splitDataToCsv(csv, ratio, val_name, train_name, seed=12345):
	np.random.seed(12345)
	df = pd.read_csv(csv)
	shuffled_df = shuffle(df)
	num_data = len(shuffled_df)
	if 0 <= ratio < 1:
		num_val = int(num_data * ratio)
		val, train = shuffled_df[:num_val], shuffled_df[num_val:]
	elif ratio <= num_data:
		val, train = shuffled_df[:ratio], shuffled_df[ratio:]
	else:
		raise ValueError("ratio must be within [0, {}] but get {}".format(num_data, ratio))
	val.to_csv(val_name, index = False)
	train.to_csv(train_name, index = False)
