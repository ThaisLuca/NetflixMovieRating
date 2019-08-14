
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import os
import pandas as pd

DATASET_SET_PATH = '/resource/NetflixShows.csv'	

def load_dataset():

	file_path = os.getcwd() + DATASET_SET_PATH

	print("Loading files...")
	dt = pd.read_csv(file_path,  delimiter = ',')

	return dt