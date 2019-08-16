
# -*- coding: utf-8 -*-

# Created by Thais Luca
# PESC/COPPE/UFRJ

import os, sys
import pandas as pd

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from preprocessing import *

TRAINING_SET_PATH = '/resource/combined_data/'	

# Lists all files in a folder
def get_list_files(path):
	return [f for f in listdir(path) if isfile(join(path, f))]

# Loads the csv file into a DataFrame
def load_dataset(path):

	file_path = os.getcwd() + path
	files = get_list_files(file_path)[:2]

	print("Loading files...")
	dt = pd.read_csv(file_path + files[1], header = None, names = ['CustomerID', 'Rating'])
	#for file in files[0:]:
		#dt.append(pd.read_csv(file_path + file,  delimiter = ','))

	dt['Rating'] = dt['Rating'].astype(float)
	dt.index = np.arange(0,len(dt))
	print('Full dataset shape: {}'.format(dt.shape))
	return dt

# Shows more information about the dataset 
def get_data_information(data):
	p = data.groupby('Rating')['Rating'].agg(['count'])

	# Counts the number of movies
	movie_count = data.isnull().sum()[1]

	# Counts the number of costumers
	cust_count = data['CustomerID'].nunique() - movie_count

	# Counts the number of ratings
	rating_count = data['CustomerID'].count() - movie_count

	ax = p.plot(kind='barh', legend=False, figsize=(15,10))
	plt.title('Total pool: {:,} Movies {:,} Customers, {:,} Ratings given'. format(movie_count, cust_count, rating_count), fontsize=20)
	plt.axis('off')

	for i in range(1,6):
		ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color='white', weight='bold')

	plt.show()