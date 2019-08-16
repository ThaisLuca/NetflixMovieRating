

# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import sys

from data import *
from preprocessing import *

TRAINING_SET_PATH = '/resource/combined_data/'	
MOVIES_FILE_PATH = '/resource/movie_titles.csv'

def main():

	# Loads dataset
	rating_data_set = load_dataset(TRAINING_SET_PATH)

	# Clean data
	rating_data_set = remove_missing_values(rating_data_set)

	# Slice data
	drop_movie_list, rating_data_set = slice_data(rating_data_set)

	# Loads movie file
	movies = load_movies_file(drop_movie_list, MOVIES_FILE_PATH)

	#print rating_data_set.head()


if __name__ == "__main__":
	sys.exit(main())