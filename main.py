# Data modelling challenge.

__author__ = 'Remus Knowles <remknowles@gmail.com>'

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

# Classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Meta Estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

F_DATA = r'data challenge test.csv'
F_CORRELATIONS = r'correlations.csv'

ENABLE_SCALING = True
PCA_COMPONENTS = 10 # None => don't do PCA

models = [
	# ('model_name', model class, shuffle)
	('SGDClassifier', SGDClassifier, True),
	('DecisionTreeClassifier', DecisionTreeClassifier, None),
	('SVC', SVC, None),
	('RandomForestClassifier', RandomForestClassifier, None),
	('AdaBoostClassifier', AdaBoostClassifier, None),
]

def main():
	df = pd.read_csv(F_DATA)
	df.drop('id', axis=1, inplace=True)

	# Split off the data we want to put in to our model.
	df = df[:60]
	df_to_predict = df[60:]

	y = df['groups']
	x = df.drop('groups', axis=1)

	# Most data is already pretty well behaved but no harm in scaling.
	if ENABLE_SCALING:
		preprocessing.scale(x)

	if PCA_COMPONENTS:
		pca = PCA(n_components=PCA_COMPONENTS)
		x = pca.fit_transform(x)

		print pca.explained_variance_
		print pca.explained_variance_ratio_

	for model_name, model, shuffle in models:
		if shuffle:
			clf = model(shuffle=True)
		else:
			clf = model()

		scores = cross_val_score(clf, x, y, cv=20)

		# 95% confidence interval for scores.
		print("%s Accuracy: %0.2f (+/- %0.2f)" % (model_name, scores.mean(), scores.std() * 2))

if __name__ == '__main__':
	main()