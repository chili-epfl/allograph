#!/usr/bin/env python
# coding: utf-8


import stroke
import learning_manager as lm
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cross_validation import train_test_split
from sklearn import svm
import math


class LetterLearnerV2:

	strokes = {}
	letters = []
	letter = None
	estimator = None
	numPointsInShapes = 70
	num_components = 0
	meanShape = []
	principleComponents = []
	parameterVariances = []
	principleValues = []
	nbClusters = 0
	
	def __init__(self, folderNames, aLetter, clusters, components):
		self.nbClusters = clusters
		self.letter = aLetter
		self.strokes = self.builStrokeCollection(folderNames, aLetter)
		self.num_components = components
		"""Get the children's strokes from the robot's"""
		for key in self.strokes:
			self.strokes[key] = stroke.childFromRobot(self.strokes[key])

		"""Build an array of all the strokes of all the children"""
		for key in self.strokes:
			for aStroke in self.strokes[key]:
				self.letters.append(stroke.strokeToArray(aStroke))
				
	def builStrokeCollection(self, folderNames, letter):
		strokes = {}
		
		for folderName in folderNames:
			for root, dirs, files in os.walk(folderName):
				for name in dirs:
					strokes[name] = lm.read_data(os.path.join(root,name),0)[letter]
		
		return strokes
		
	def clusterize(self):
		self.estimator = KMeans(n_clusters=self.nbClusters,init='k-means++')
		self.estimator.fit(self.letters)
		
		
	def classify(self):
		X_train, X_test, y_train, y_test = train_test_split(np.array(self.letters,dtype=np.float64),self.estimator.labels_)
		clf = svm.SVC()
		clf.fit(X_train, y_train) 
		labelsPredicted =  clf.predict(X_test)
    
		diff = [predicted-real for predicted,real in zip(labelsPredicted, y_test)]
    
		count = 0
		for d in diff:
			if (d == 0):
				count = count + 1	
			
		accuracy = (float(count)/len(diff))*100
		print "accuracy in %:"
		print accuracy 
		return clf
		
	def clfPredict(self, clf, aStroke):
		return clf.predict(stroke.strokeToArray(aStroke).reshape(1,-1))
		
	def performPCA(self):
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.letters)
			dataMat = np.array(filteredLetters).reshape((len(filteredLetters), self.numPointsInShapes*2))
			covarMat = np.cov(dataMat.T)
			eigVals, eigVecs = np.linalg.eig(covarMat)
			self.principleComponents.append(np.real(eigVecs[:, 0:self.num_components]))
			self.principleValues.append(np.real(eigVals[0:self.num_components]))
			self.parameterVariances.append(np.real(eigVals[0:self.num_components]))
			self.meanShape.append(dataMat.mean(0).reshape((self.numPointsInShapes * 2, 1)))
			i += 1
			
	def __project(self, letter, index):
		return np.dot(self.principleComponents[index].T, (letter.reshape(-1, 1) - self.meanShape[index])).reshape(self.num_components,)
	
	def __projectBack(self, letter, index):
		return (self.meanShape[index] + np.dot(self.principleComponents[index], letter).reshape(-1, 1)).reshape(self.numPointsInShapes*2, )
	
	def _projectCentroids(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components))
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			projected[i] = self.__project(aCentroid, i)
			i = i + 1
		return projected		
	
	def modifyCoordinates(self, label, letter, factor):
		coordinates = self.__project(letter, label)
		coordinates[0] += factor
		return self.__projectBack(coordinates, label)
		
	def printLetter(self, letter):
		stroke.arrayToStroke(letter).plot()
    
    #first in blue, second in red
	def printLetters(self, letters):
		stroke.plot_list(map(lambda l: stroke.arrayToStroke(l), letters))
		
	def getEstimator(self):
		return self.estimator
