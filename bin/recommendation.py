
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

from sklearn.neighbors import NearestNeighbors
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import train_test_split
from surprise import accuracy


# Males recommendation by user
def recommend_by_user(user, data, movies, drop_movie_list):

	return recommend_using_svd(data, user, movies, drop_movie_list)


# Recommendation using SVD
def recommend_using_svd(data, user, movies, drop_movie_list):
	reader = Reader()

	data = Dataset.load_from_df(data[['CustomerID', 'MovieID', 'Rating']][:100000], reader)
	trainset = data.build_full_trainset()

	svd = SVD()
	svd.train(trainset)

	user_1 = movies.copy()
	user_1 = user_1.reset_index()
	user_1 = user_1[~user_1['MovieID'].isin(drop_movie_list)]

	print("Recommending movies using SVD...")
	user_1['Estimate_Score'] = user_1['MovieID'].apply(lambda x: svd.predict(user, x).est)

	user_1 = user_1.drop('MovieID', axis=1)

	user_1 = user_1.sort_values('Estimate_Score', ascending=False)
	return user_1[:10]

# Tests SVD accuracy
def test_svd(data):
	reader = Reader()

	data = Dataset.load_from_df(data[['CustomerID', 'MovieID', 'Rating']][:100000], reader)
	trainset, testset = train_test_split(data, test_size=.25)

	svd = SVD()
	svd.fit(trainset)
	predictions = svd.test(testset)

	accuracy.rmse(predictions)