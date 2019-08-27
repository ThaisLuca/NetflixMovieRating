

# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import sys

from data import *
from metrics import *
from preprocessing import *
from recommendation import *
from collections import defaultdict
from surprise.model_selection import KFold
from surprise import Reader, Dataset, SVD, KNNBaseline, KNNBasic

TRAINING_SET_PATH = '/resource/combined_data/'	
MOVIES_FILE_PATH = '/resource/movie_titles.csv'
N = 10

def main():

	# Loads dataset
	rating_data_set = load_dataset(TRAINING_SET_PATH)

	# # Clean data
	rating_data_set = remove_missing_values(rating_data_set)

	# Slice data
	drop_movie_list, rating_data_set = slice_data(rating_data_set)

	# Loads movie file
	movies = load_movies_file(drop_movie_list, MOVIES_FILE_PATH)


	reader = Reader()

	sim_options = {'name': 'cosine', 'min_support': 2, 'shrinkage': 100, 'user_based': True}
	bsl_options = {'method': 'sgd'}
	data = Dataset.load_from_df(rating_data_set[['CustomerID', 'MovieID', 'Rating']][:1000], reader)

	kf = KFold(n_splits=5)
	#algo = SVD()
	algo = KNNBaseline(k=N, sim_options=sim_options, bsl_options=bsl_options)

	i = 0

	for trainset, testset in kf.split(data):
		print("Running fold: ", i)
		algo.fit(trainset)
		predictions = algo.test(testset)
		precisions, recalls = precision_recall(predictions, 20)

	    # Precision and recall can then be averaged over all users
		print(sum(prec for prec in precisions.values()) / len(precisions))
		print(sum(rec for rec in recalls.values()) / len(recalls))

		i += 1


if __name__ == "__main__":
	sys.exit(main())