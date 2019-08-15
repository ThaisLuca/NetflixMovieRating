
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import os, sys
import pandas as pd

from os import listdir
from os.path import isfile, join

TRAINING_SET_PATH = '/resource/combined_data/'	

# Lists all files in a folder
def get_list_files(path):
	return [f for f in listdir(path) if isfile(join(path, f))]

# Loads the csv file into a DataFrame
def load_dataset(path):

	file_path = os.getcwd() + path
	files = get_list_files(file_path)[:2]

	print("Loading files...")
	dt = pd.read_csv(file_path + files[1], header = None, names = ['CustomerID', 'Rating'])
	for file in files[0:]:
		dt.append(pd.read_csv(file_path + file,  delimiter = ','))

	dt['Rating'] = dt['Rating'].astype(float)
	print("%d movies." % dt.size)
	print("The shape of the dataframe is {}".format(dt.shape))
	return dt
