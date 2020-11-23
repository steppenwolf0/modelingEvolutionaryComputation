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

# this is an incredibly useful function
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
directory="best"

def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./"+directory+"/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./"+directory+"/labels.csv", header=None)
	dfFeats = read_csv("./"+directory+"/features_0.csv", header=None)
	dfClasses = read_csv("./"+directory+"/classes.csv", header=None)		
	return dfData.values, dfLabels.values.ravel(), dfFeats.values.ravel(), dfClasses.values.ravel() # to have it in the format that the classifiers like


def fakeBootStrapper2():

	
	X,Y, feats, classes=loadDataset()
	#scaler = StandardScaler()
	#X = scaler.fit_transform(X)
	labels=np.max(Y)+1
	
	xClass=[]
	
	for j in range(0,labels):
		temp=[]
		for i in range(0,len(Y)):
			if (Y[i]==j):
				values=[]
				values=X[i,:]
				temp.append(values)
		xClass.append(temp)

	
	green_diamond = dict(markerfacecolor='#0335fc', marker='D')
	red_square = dict(markerfacecolor='#03bafc', marker='s')
	
	axisVal=[]
	for i in range (1,len(xClass[0][0])+1):
		axisVal.append(i)
	fig1, ax1 = plt.subplots()
	for j in range(0,labels):
	
		
		
		#ax1.set_title(classes[j])
		if (j==0):
			
			
			bp=ax1.boxplot(np.array(xClass[j]),0,'', patch_artist=True) #No Outliers
			#bp=ax1.boxplot(np.array(xClass[j]), patch_artist=True, flierprops=green_diamond) #With Outliers
			## change color and linewidth of the medians
			for median in bp['medians']:
				median.set(color='#b2df8a', linewidth=2)
			for box in bp['boxes']:
				# change outline color
				box.set( color='#7570b3', linewidth=2)
				# change fill color
				box.set( facecolor = '#0335fc' )
		if (j==1):
			#bp=ax1.boxplot(np.array(xClass[j]), patch_artist=True, flierprops=red_square, notch=True) #With Outliers
			bp=ax1.boxplot(np.array(xClass[j]),0,'', patch_artist=True)  #No Outliers
		
			## change color and linewidth of the medians
			for median in bp['medians']:
				median.set(color='#ffffff', linewidth=2)
			for box in bp['boxes']:
				# change outline color
				box.set( color='#7570b3', linewidth=2)
				# change fill color
				box.set( facecolor = '#03bafc' )
		if (j==2):
			#bp=ax1.boxplot(np.array(xClass[j]), patch_artist=True, flierprops=red_square, notch=True) #With Outliers
			bp=ax1.boxplot(np.array(xClass[j]),0,'', patch_artist=True)  #No Outliers
		
			## change color and linewidth of the medians
			for median in bp['medians']:
				median.set(color='#33ffcc', linewidth=2)
			for box in bp['boxes']:
				# change outline color
				box.set( color='#7570b3', linewidth=2)
				# change fill color
				box.set( facecolor = '#cc33ff' )
		
		labels = feats
		patch_0 = mpatches.Patch(color='#0335fc', label=classes[0])
		patch_1 = mpatches.Patch(color='#03bafc', label=classes[1])
		#patch_2 = mpatches.Patch(color='#cc33ff', label=classes[2])
		plt.legend(handles=[patch_0,patch_1])
		plt.xticks(axisVal, labels, rotation='vertical')
		plt.subplots_adjust(bottom=0.30)
		ax1.tick_params(axis='both', which='major', labelsize=10)
		plt.xlabel("Variables")
		plt.ylabel("Expression Levels")
	
	plt.show()
	return

if __name__ == "__main__" :
	sys.exit( fakeBootStrapper2() )