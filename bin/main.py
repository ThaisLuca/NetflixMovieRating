

# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import sys

from data import *
from metrics import *
from preprocessing import *
from recommendation import *
from surprise.model_selection import KFold

TRAINING_SET_PATH = '/resource/combined_data/'	
MOVIES_FILE_PATH = '/resource/movie_titles.csv'
N = 10

def main():

	# Loads dataset
	rating_data_set = load_dataset(TRAINING_SET_PATH)

	# # Clean data
	rating_data_set = remove_missing_values(rating_data_set)

	# # Slice data
	drop_movie_list, rating_data_set = slice_data(rating_data_set)

	# Loads movie file
	movies = load_movies_file(drop_movie_list, MOVIES_FILE_PATH)

	# Get movie x user matrix
	#rating_data_set = format_data_pivot_table(rating_data_set)

	# Test SVD
	#test_svd(rating_data_set)

	# Test KNN
	#test_KNN(rating_data_set)


	predictions = recommend_using_knn(rating_data_set, N)
	#top_n = get_top_n(predictions, N)
	precision, recall = precision_recall(predictions, 10, 4)
	print("Precision: ", get_average(precision))
	print("Recall: ", get_average(recall))
	#show_user_top_n(top_n[2337449], movies)

	return

	reader = Reader()

	sim_options = {'name': 'pearson_baseline', 'user_based': True}
	data = Dataset.load_from_df(rating_data_set[['CustomerID', 'MovieID', 'Rating']][:100000], reader)

	kf = KFold(n_splits=5)
	#algo = SVD()
	algo = KNNBaseline(k=N, sim_options=sim_options)

	i = 0

	for trainset, testset in kf.split(data):
		print("Running fold: ", i)
		algo.fit(trainset)
		predictions = algo.test(testset)
		precisions, recalls = precision_recall(predictions, 10)

	    # Precision and recall can then be averaged over all users
		print(sum(prec for prec in precisions.values()) / len(precisions))
		print(sum(rec for rec in recalls.values()) / len(recalls))

		i += 1

	return


if __name__ == "__main__":
	sys.exit(main())