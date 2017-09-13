# Data modelling challenge.

__author__ = 'Remus Knowles <remknowles@gmail.com>'

import pandas as pd

F_DATA = r'data challenge test.csv'
F_CORRELATIONS = r'correlations.csv'

def main():
	df = pd.read_csv(F_DATA)
	df.drop('id', axis=1, inplace=True)

	# Split off the data we want to put in to our model.
	df = df[:60]
	df_to_predict = df[60:]

	# Check for any obvious correlation in explanatory variables.
	print df.corr().to_csv(F_CORRELATIONS)

if __name__ == '__main__':
	main()