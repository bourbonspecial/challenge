# Data modelling challenge.

__author__ = 'Remus Knowles <remknowles@gmail.com>'

import pandas as pd

F_DATA = r'data challenge test.csv'

def main():
	df = pd.read_csv(F_DATA)

	print df.head()

if __name__ == '__main__':
	main()