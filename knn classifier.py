#!/usr/bin/env python
"""This code analyses the classification of a given dataset based on KNN Classifier"""

import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler

# Get project root path
rootDirectory = os.path.dirname(os.path.abspath(__file__))
# Get Input File Details
inputFile = rootDirectory + '\input\wine.csv'
outputPath = rootDirectory + 'output'
# Read input CSV File
wineData = pd.read_csv(inputFile)
# Get All Column's name of the input file
columnHeaders = list(wineData.columns.values)
# Create all features to be considered for classification
featureColumns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',\
		  'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#'quality' is the class attribute we are predicting
classColumn = 'quality'
wineFeature = wineData[featureColumns]
wineClass = wineData[classColumn]

# Splitting given dataset - Train: 75%, Test: 25%
trainFeature, testFeature, trainClass, testClass = train_test_split(wineFeature, wineClass, stratify=wineClass, train_size=.75, test_size=.25)
trainAccuracy = []
testAcuracy = []
# Using KNeighborsClassifier with 5 neighbours and 'brute' algorithm, distance metric = 3
knn = KNeighborsClassifier(n_neighbors=5, p=3, algorithm='brute', metric='minkowski')
knn.fit(trainFeature, trainClass)
# Printing test and train accuracy of the classifier
print("Train set accuracy: {:.2f}".format(knn.score(trainFeature, trainClass)))
print("Test set accuracy: {:.2f}".format(knn.score(testFeature, testClass)))
# Confusion matrix(6x6) including 'All' for test data
prediction = knn.predict(testFeature)
print("Confusion matrix:")
print(pd.crosstab(testClass, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

# Application of 10-fold stratified cross-validation
crossValidationScore = cross_val_score(knn, wineFeature, wineClass, cv=10)
print("Cross-validation scores: {}".format(crossValidationScore))
print("Average cross-validation score: {:.2f}".format(crossValidationScore.mean()))

# Pre-processing of Dataset
wineFeatureFinal = preprocessing.maxabs_scale(wineFeature)
wineFeatureFinal2 = MaxAbsScaler().fit_transform(wineFeatureFinal)

cvScoreFinal = cross_val_score(knn, wineFeatureFinal, wineClass, cv=10)
print("Post Pre-Processing Cross-validation scores: {}".format(cvScoreFinal))
print("Post Pre-Processing Average cross-validation score: {:.2f}".format(cvScoreFinal.mean()))
