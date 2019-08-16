
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import pandas as pd
import numpy as np

# Removes missing values such as null and NaN
def remove_missing_values(data):
	print("Cleaning data..")
	print("Adjusting movies id.")
	dt_nan = pd.DataFrame(pd.isnull(data.Rating))
	dt_nan = dt_nan[dt_nan['Rating'] == True]
	dt_nan = dt_nan.reset_index()

	movies = []
	movie_id = 1

	for i,j in zip(dt_nan['index'][1:], dt_nan['index'][:-1]):
		temp = np.full((1,i-j-1), movie_id)
		movies = np.append(movies, temp)
		movie_id += 1

	last_record = np.full((1,len(data) - dt_nan.iloc[-1,0] - 1), movie_id)
	movies = np.append(movies, last_record)

	print('Movie array: {}'.format(movies))
	print('Length: {}'.format(len(movies)))

	print("Removing missing values.")
	data = data[pd.notnull(data['Rating'])]
	data['MovieID'] = movies.astype(int)
	data['CustomerID'] = data['CustomerID'].astype(int)	
	#print('Dataset Examples')
	#print(data.iloc[::5000000,:])

	return data

# Removes unpopular movies (with too less reviews)
# Removes inactive customers (who give too less reviews)
def slice_data(data):
	f = ['count', 'mean']
	dt_movie_summary = data.groupby('MovieID')['Rating'].agg(f)
	dt_movie_summary.index = dt_movie_summary.index.map(int)
	movie_brenchmark = round(dt_movie_summary['count'].quantile(0.7), 0)
	drop_movie_list = dt_movie_summary[dt_movie_summary['count'] < movie_brenchmark].index
	print('Movie minimum times of review: {}'.format(movie_brenchmark))

	dt_cust_summary = data.groupby('CustomerID')['Rating'].agg(f)
	dt_cust_summary.index = dt_cust_summary.index.map(int)
	cust_brenchmark = round(dt_cust_summary['count'].quantile(0.7), 0)
	drop_cust_list = dt_cust_summary[dt_cust_summary['count'] < cust_brenchmark].index

	print('Customer minimum times of review: {}'.format(cust_brenchmark))

	print('Original Shape: {}'.format(data.shape))
	data = data[~data['MovieID'].isin(drop_movie_list)]
	data = data[~data['CustomerID'].isin(drop_cust_list)]
	print('After trim shape: {}'.format(data.shape))
	print('Data Examples')
	print(data.iloc[::5000000, :])
