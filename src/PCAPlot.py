# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
# this is an incredibly useful function
from pandas import read_csv
from sklearn.manifold import TSNE

def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./best/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./best/labels.csv", header=None)
		
	return dfData.values, dfLabels.values.ravel() # to have it in the format that the classifiers like


def runFeatureReduce() :

	print("Loading dataset...")
	X, y = loadDataset()
	target_names=[]
	target_names.append('Healthy Lung')
	target_names.append('COVID-19')
	
	#scaler = StandardScaler()
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	
	from sklearn.decomposition import PCA
	from sklearn.decomposition import KernelPCA
	from sklearn.manifold import SpectralEmbedding
	transformer = KernelPCA(n_components=4, kernel='linear')
	X_r = transformer.fit_transform(X)
	#pca = PCA(n_components=2)
	#pca = PCA(n_components=2)
	#X_r = pca.fit(X).transform(X)
	#tsne = TSNE(n_components=2, init='pca', perplexity=100)
	#embedding = SpectralEmbedding(n_components=2)
	#X_r = embedding.fit_transform(X)
	
	import matplotlib.pyplot as plt
	# Percentage of variance explained for each components


	plt.figure()
	N = 7
	r = 2 * np.random.rand(N)
	theta = 2 * np.pi * np.random.rand(N)
	colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple', 'green', 'grey']
	lw = 2

	for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6 ], target_names):
		plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('')
	plt.show()
	#plt.savefig(name+".png")
	
	return

if __name__ == "__main__" :
	sys.exit( runFeatureReduce() )