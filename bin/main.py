

# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import sys

from data import *
from preprocessing import *
from recommendation import *

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

	# Get movie x user matrix
	#rating_data_set = format_data_pivot_table(rating_data_set)

	recommendations = recommend_by_user(785314, rating_data_set, movies, drop_movie_list)
	print recommendations, type(recommendations)
	print("Precision @ N")
	movies_rated_by_user = rating_data_set[(rating_data_set['CustomerID'] == 785314)]
	correct_predicted = recommendations.isin(movies_rated_by_user).count()
	print correct_predicted
	print(correct_predicted / recommendations)


if __name__ == "__main__":
	sys.exit(main())