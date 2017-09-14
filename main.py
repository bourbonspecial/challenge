# Data modelling challenge.

__author__ = 'Remus Knowles <remknowles@gmail.com>'

import csv
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

# Classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Ensemble Classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

F_DATA = r'data challenge test.csv'
F_CORRELATIONS = r'correlations.csv'
F_OUT = r'models.csv'

ENABLE_SCALING = True
PCA_COMPONENTS = 11 # None => don't do PCA
CROSS_VALIDATION_ITERATIONS = 5

models = [
	# ('model_name', model class, shuffle, voting)
	('GaussianProcessClassifier', GaussianProcessClassifier, True),
	('KNeighborsClassifier', KNeighborsClassifier, True),
	('DecisionTreeClassifier', DecisionTreeClassifier, True),
	('SVC', SVC, True),
	('NuSVC', NuSVC, True),
	('LinearSVC', LinearSVC, True),
	# Ensemble classifiers
	('AdaBoostClassifier', AdaBoostClassifier, True),
	('GradientBoostingClassifier', GradientBoostingClassifier, True),
	('RandomForestClassifier', RandomForestClassifier, True),
	('VotingClassifier', VotingClassifier, False),
]

def main():
	df = pd.read_csv(F_DATA)
	df.drop('id', axis=1, inplace=True)

	y = df['groups'][:60]
	df = df.drop('groups', axis=1)

	# Most data is already pretty well behaved but no harm in scaling.
	if ENABLE_SCALING:
		preprocessing.scale(df)

	if PCA_COMPONENTS:
		pca = PCA(n_components=PCA_COMPONENTS)
		df = pca.fit_transform(df)

	# Split off the data we want to put in to our model.
	x_predict = df[60:]
	x = df[:60]

	clf = GradientBoostingClassifier()
	clf.fit(x,y)
	print clf.predict(x_predict)

if __name__ == '__main__':
	main()