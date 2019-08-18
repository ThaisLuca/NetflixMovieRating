
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

from surprise import Reader, Dataset, SVD, evaluate, KNNBaseline
from surprise.model_selection import cross_validate
import numpy as np

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

# Recommendation using KNN as learning algorithm
def recommend_using_knn(data, N):
	reader = Reader()

	sim_options = {'name': 'pearson_baseline', 'shrinkage': 10, 'min_support': 10, 'user_based': True}

	data = Dataset.load_from_df(data[['CustomerID', 'MovieID', 'Rating']][:100000], reader)
	trainset = data.build_full_trainset()

	knn = KNNBaseline(k=N, sim_options=sim_options)
	knn.fit(trainset)

	testset = trainset.build_anti_testset()
	print("Test set", len(testset))
	predictions = knn.test(testset)

	return predictions


# Tests SVD accuracy
def test_svd(data):
	reader = Reader()

	data = Dataset.load_from_df(data[['CustomerID', 'MovieID', 'Rating']][:100000], reader)

	svd = SVD()
	cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


def recommend_knn(user, data, N):
	kdt = KDTree(data, leaf_size=30, metric='euclidean')
	return kdt.query(user, k=data.shape[1], return_distance=False)

# Tests KNN accuracy
def test_KNN(data):
	reader = Reader()

	data = Dataset.load_from_df(data[['CustomerID', 'MovieID', 'Rating']][:100000], reader)

	knn = KNNBasic()
	cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)