
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, SVD


# Males recommendation by user
def recommend_by_user(user, data, movies, drop_movie_list, N):

	return recommend_using_svd(data, user, movies, drop_movie_list, N)


# Recommendation using SVD as learning algorithm
def recommend_using_svd(data):
	reader = Reader()

	data = Dataset.load_from_df(data[['CustomerID', 'MovieID', 'Rating']][:100000], reader)

	trainset = data.build_full_trainset()

	svd = SVD()
	svd.fit(trainset)

	testset = trainset.build_anti_testset()
	predictions = svd.test(testset)

	return predictions

# Recommendation using KNN as learning algorithm for movie movie similarity (User_based = True)
def recommend_using_knn(data, N):
	users_neighbors = defaultdict()

	trainset, testset = train_test_split(data, test_size=0.3)

	knn = NearestNeighbors(n_neighbors=N, algorithm='ball_tree')
	knn.fit(trainset)

	print("Test set", len(testset))
	predictions = knn.kneighbors(testset, return_distance=False)

	i = 0
	for index, row in testset.iterrows():
		users_neighbors[row['CustomerID']] = predictions[i]
		i += 1

	return users_neighbors

def recommend_knn(user, data, N):
	kdt = KDTree(data, leaf_size=30, metric='euclidean')
	return kdt.query(user, k=data.shape[1], return_distance=False)

def select_movies_knn(predictions, data):
	movies_per_user = defaultdict()

	for key, val in predictions.items():
		neighbors = data[data['CustomerID'].isin(val)]
		movies_seen = data[data['CustomerID'].isin([key])]
		movies_per_user[key] = data[~data['MovieID'].isin(movies_seen)]

		print "neighbors", neighbors
		print "movies_seen", movies_seen
		print "movies_per_user", movies_per_user[key]

		break
		
	print movies_per_user
	return movies_per_user
