
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

from __future__ import division
from collections import defaultdict
import pandas as pd

def precision_recall(predictions, N, threshold=3.5):

	user_est_true = defaultdict(list)

	# First map the predictions to each user
	for uid, _, true_r, est, _ in predictions:
		user_est_true[uid].append((est, true_r))

	precisions = dict()
	recalls = dict()
	for uid, user_ratings in user_est_true.items():

		# Sort user ratings by estimated value
		user_ratings.sort(key=lambda x: x[0], reverse=True)

		# Number of relevant items
		n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

		# Number of recommended items in top n
		n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:N])

		# Number of relevant and recommeded items in top n
		n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:N])

		# Precision@N: Proportion of recommended items that are relevant
		precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

		# Recall@N: Proportion of relevant items that are recommended
		recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

	return precisions, recalls


def get_average(measure):
	return sum(mes for mes in measure.values()) / len(measure)

# Returns Precision on Top N
def precision(recommended, testing):
	df_intersection = recommended.MovieID.isin(testing.MovieID).count()
	
	return (df_intersection / len(recommended.index))

# Returns Recall on Top N
def recall(answers, y, N):
	df_intersection = recommended.MovieID.isin(testing.MovieID).count()

	return (df_intersection / len(testing.index))


# Get Top N movies recommended for each user
def get_top_n(predictions, n):
	top_n = defaultdict(list)

	# First map the predicitions to each user
	for uid, iid, true_r, est, _ in predictions:
		top_n[uid].append((iid, est))

	# Then sort the predictions for each user and retrieve the k highest ones.
	for uid, user_ratings in top_n.items():
		user_ratings.sort(key=lambda x: x[1], reverse=True)
		top_n[uid] = user_ratings[:n]

	return top_n

# Print the recommended items for each user
def show_all_users_top_n(top_n):
	for uid, user_ratings in top_n.items():
		print(uid, [iid for (iid, _) in user_ratings])


def show_user_top_n(top_n_user, movies):
	movies_for_user = []
	for movie in top_n_user:
		movies_for_user.append(movies[movies.index == movie[0]])

	print movies_for_user