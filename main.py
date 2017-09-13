# Data modelling challenge.

__author__ = 'Remus Knowles <remknowles@gmail.com>'

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

# Classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Meta Estimators
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

F_DATA = r'data challenge test.csv'
F_CORRELATIONS = r'correlations.csv'

ENABLE_SCALING = True
PCA_COMPONENTS = None # None => don't do PCA
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

	for model_name, model, voting in models:
		# Bit hacky, bescause we could have a voting classifier here.
		# Just so happens that we don't at the moment.
		if voting:
			clf = model()
			scores = cross_val_score(clf, x, y, cv=CROSS_VALIDATION_ITERATIONS)
			# 95% confidence interval for scores.
			print("%s Accuracy: %0.2f (+/- %0.2f)" % (model_name, scores.mean(), scores.std() * 2))

	# Need to do voting classifier seperately, as func signature is slightly different.
	voting_clfs = [(model_name,model()) for model_name,model,voting in models if voting]
	clf = VotingClassifier(estimators=voting_clfs,voting='hard')
	scores = cross_val_score(clf, x, y, cv=CROSS_VALIDATION_ITERATIONS)
	print("%s Accuracy: %0.2f (+/- %0.2f)" % ('VotingClassifier', scores.mean(), scores.std() * 2))

if __name__ == '__main__':
	main()