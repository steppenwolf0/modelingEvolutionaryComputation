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
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from scipy import stats
# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# this is an incredibly useful function
from pandas import read_csv
from scipy import stats

def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./best/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./best/labels.csv", header=None)
		
	return dfData.values, dfLabels.values.ravel() # to have it in the format that the classifiers like


def runFeatureReduce() :
	
	orig_stdout = sys.stdout
	f = open('./best/manova.txt', 'w')
	sys.stdout = f
	
	print("Loading dataset...")
	X, y = loadDataset()
	
	maov = MANOVA(X,y)
	
	
	print(len(X))
	print(len(X[0]))
	print(len(y))

	print(maov.mv_test())
	
	est = sm.OLS(y, X)
	est2 = est.fit()
	print(est2.summary())
	
	cases=[]
	controls=[]
	for i in range (0,len(y)):
		valuesTemp=[]
		for j in range (0,len(X[0])):
			valuesTemp.append(X[i,j])
		if(y[i]==0):
			controls.append(valuesTemp)
		else:
			cases.append(valuesTemp)
	
	controls=np.asarray(controls)
	cases=np.asarray(cases)
	
	ttest,pval =  stats.f_oneway(controls,cases)
	print("p-value ANOVA",pval)
	pd.DataFrame(pval).to_csv("./pANOVA.csv", header=None, index =None)
	
	
	ttest,pval =  stats.ttest_ind(controls,cases)
	print("p-value Two sampled T-test",pval)
	pd.DataFrame(pval).to_csv("./pttestInd.csv", header=None, index =None)

	meanControls=np.mean(controls, axis=0)
	print(meanControls)
	pd.DataFrame(meanControls).to_csv("./meanControls.csv", header=None, index =None)
	
	meanCases=np.mean(cases, axis=0)
	print(meanCases)
	pd.DataFrame(meanCases).to_csv("./meanCases.csv", header=None, index =None)
	
	sys.stdout = orig_stdout
	f.close()
	return

if __name__ == "__main__" :
	sys.exit( runFeatureReduce() )