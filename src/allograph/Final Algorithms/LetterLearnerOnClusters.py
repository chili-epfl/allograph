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


class LetterLearnerOnClusters:

	strokes = {}
	letters = []
	letter = None
	estimator = None
	clf = None
	numPointsInShapes = 70
	num_components = 0
	meanShape = []
	principleComponents = []
	parameterVariances = []
	principleValues = []
	nbClusters = 0
	X_train = []
	X_test = []

	
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
				
		"""splits the sets of letters in two sets: one to train, the other to test."""	
		self.X_train, self.X_test = train_test_split(np.array(self.letters,dtype=np.float64), test_size=0.1)
		"""Saves the number of samples"""
		self.numShapesInDataset = len(self.X_train)
				
	"""Builds the collection of strokes from list of root folders."""		
	def builStrokeCollection(self, folderNames, letter):
		strokes = {}
		
		for folderName in folderNames:
			for root, dirs, files in os.walk(folderName):
				for name in dirs:
					strokes[name] = lm.read_data(os.path.join(root,name),0)[letter]
		
		return strokes
		
	"""Clusterizes using k-means and a number of clusters specified during the instanciation of the class."""
	def clusterize(self):
		self.estimator = KMeans(n_clusters=self.nbClusters,init='k-means++')
		self.estimator.fit(self.X_train)
	
	"""Classifies using SVM using the training set"""
	def classify(self):
		self.clf = svm.SVC()
		self.clf.fit(self.X_train, self.estimator.labels_)
		
		
	"""Predicts a letter label"""	
	def predict(self, letter):
		return self.clf.predict(letter)
		
	"""Perform PCA on each clusters"""
	def performPCA(self):
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.X_train)
			dataMat = np.array(filteredLetters).reshape((len(filteredLetters), self.numPointsInShapes*2))
			covarMat = np.cov(dataMat.T)
			eigVals, eigVecs = np.linalg.eig(covarMat)
			self.principleComponents.append(np.real(eigVecs[:, 0:self.num_components]))
			self.principleValues.append(np.real(eigVals[0:self.num_components]))
			self.parameterVariances.append(np.real(eigVals[0:self.num_components]))
			self.meanShape.append(dataMat.mean(0).reshape((self.numPointsInShapes * 2, 1)))
			i += 1
	
	"""Helper function to project a letter on the eigen space of the specified cluster"""
	def __project(self, letter, index):
		return np.dot(self.principleComponents[index].T, (letter.reshape(-1, 1) - self.meanShape[index])).reshape(self.num_components,)
	
	"""Helper function to project a letter back to the regular space"""
	def __projectBack(self, letter, index):
		return (self.meanShape[index] + np.dot(self.principleComponents[index], letter).reshape(-1, 1)).reshape(self.numPointsInShapes*2, )
	
	"""Helper funciton to project all the centroids to the eigen space"""
	def _projectCentroids(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components))
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			projected[i] = self.__project(aCentroid, i)
			i = i + 1
		return projected		
	
	"""Projects onto the eigenspace modify the first coordinate and project back onto the regular space"""
	def _modifyCoordinates(self, label, letter, factor):
		coordinates = self.__project(letter, label)
		coordinates[0] += factor
		return self.__projectBack(coordinates, label)
	
	"""modify all the letters of the test set and return them"""	
	def testAlgo(self, factor):
		ls = []
		for aLetter in self.X_test:
			label = self.predict(np.array(aLetter).reshape(1, -1))[0]
			ls.append(self._modifyCoordinates(label, aLetter, factor))
		return ls
		
	def printLetter(self, letter):
		stroke.arrayToStroke(letter).plot()
    
    #first in blue, second in red
	def printLetters(self, letters):
		stroke.plot_list(map(lambda l: stroke.arrayToStroke(l), letters))
		
	def getEstimator(self):
		return self.estimator
