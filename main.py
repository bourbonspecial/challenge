# Data modelling challenge.

__author__ = 'Remus Knowles <remknowles@gmail.com>'

import pandas as pd

from sklearn import preprocessing

F_DATA = r'data challenge test.csv'
F_CORRELATIONS = r'correlations.csv'

def main():
	df = pd.read_csv(F_DATA)
	df.drop('id', axis=1, inplace=True)

	# Split off the data we want to put in to our model.
	df = df[:60]
	df_to_predict = df[60:]

	y = df['groups']
	x = df.drop('groups', axis=1)

	# Most data is already pretty well behaved but no harm in scaling.
	preprocessing.scale(x)

	x_train = x[:50]
	x_test = x[50:]
	y_train = y[:50]
	y_test = y[50:]

	

if __name__ == '__main__':
	main()