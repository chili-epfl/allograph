#!/usr/bin/env python
# coding: utf-8


import stroke
import learning_manager as lm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cross_validation import train_test_split
from sklearn import svm
import math


class LetterLearner:
	strokes = {}
	letters = []
	children = []
	letter = None
	estimator = None
	clf = None
	numShapesInDataset = 0
	numPointsInShapes = 70
	num_components = 0
	meanShape = 0
	principleComponents = None
	parameterVariances = None
	principleValues = None
	X_train = None
	X_test = None
	children_train = []
	children_test = []
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
				self.children.append(key)
				
		self.X_train, self.X_test, self.children_train, self.children_test = train_test_split(np.array(self.letters,dtype=np.float64), np.array(self.children, dtype=np.str_))
		self.numShapesInDataset = len(self.X_train)
		
				
	def builStrokeCollection(self, folderNames, letter):
		strokes = {}
		
		for folderName in folderNames:
			for root, dirs, files in os.walk(folderName):
				for name in dirs:
					strokes[name] = lm.read_data(os.path.join(root,name),0)[letter]
		
		return strokes
		
	def clusterize(self):
		self.estimator = KMeans(n_clusters=self.nbClusters,init='k-means++')
		self.estimator.fit(self.X_train)
	
	def childrenNamePerCluster(self):
		i = 0
		for aCluster in self.estimator.cluster_centers_:
			j = 0
			for let in self.X_train:
				if (self.estimator.predict(np.array(let).reshape(1,-1)) == i):
					print i, ": ", self.children_train[j]
				j += 1
			i += 1
		
	def classify(self):
		self.clf = svm.SVC()
		self.clf.fit(self.X_train, self.estimator.labels_)
		
	def predict(self, letter):
		return self.clf.predict(letter)
		
		
	def performPCA(self):
		dataMat = np.array(self.X_train).reshape((self.numShapesInDataset, self.numPointsInShapes*2))
		covarMat = np.cov(dataMat.T)
		eigVals, eigVecs = np.linalg.eig(covarMat)
		self.principleComponents = np.real(eigVecs[:, 0:self.num_components])
		self.principleValues = np.real(eigVals[0:self.num_components])
		self.parameterVariances = np.real(eigVals[0:self.num_components])
		self.meanShape = dataMat.mean(0).reshape((self.numPointsInShapes * 2, 1))
		
		
	
	def __project(self, letter):
		return np.dot(self.principleComponents.T, (letter.reshape(-1, 1) - self.meanShape)).reshape(self.num_components,)
	
	def __projectBack(self, letter):
		return (self.meanShape + np.dot(self.principleComponents, letter).reshape(-1, 1)).reshape(self.numPointsInShapes*2, )
		
	def _projectCentroids(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components))
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			projected[i] = self.__project(aCentroid)
			i = i + 1
		return projected
		
	def _projectClustersV1(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components, 3))
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.X_train)
			projectedLetters = map(lambda letter: self.__project(letter), filteredLetters)
			
			tuples = []
			for j in range(self.num_components):
				mean = 0
				for letter in projectedLetters:
					mean = mean + letter[j]
				mean = mean/len(projectedLetters)
				tuples.append((mean - 2*(math.sqrt(self.parameterVariances[j])/math.sqrt(len(projectedLetters))), mean + 2*(math.sqrt(self.parameterVariances[j])/math.sqrt(len(projectedLetters)))))
			finalTuples = map(lambda tup: (tup[0][0], tup[1], tup[0][1]), zip(tuples, self._projectCentroids()[i]))
			projected[i] = np.array(finalTuples)
			i = i + 1
		return projected
		
	def _projectClustersV2(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components, 2))
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.X_train)
			projectedLetters = map(lambda letter: self.__project(letter), filteredLetters)
			varNormalized = []
			for j in range(self.num_components):
				varNormalized.append(np.var(map(lambda x: x[j], projectedLetters))/self.principleValues[j])
			finalTuples = zip(varNormalized, self._projectCentroids()[i])
			projected[i] = np.array(finalTuples)
			i = i + 1
		return projected
	
	def getMoreImportantDimensionV1(self):
		tuples = []
		for cluster in self._projectClustersV1():
			dim = -1
			diff = np.Infinity
			for i in range(self.num_components):
				if (diff > (cluster[i][2]-cluster[i][0])):
					diff = cluster[i][2]-cluster[i][0]
					dim = i
			tuples.append((cluster[dim], dim))
		return tuples
		
	def getMoreImportantDimensionV2(self):
		tuples = []
		for cluster in self._projectClustersV2():
			dim = -1
			diff = np.Infinity
			for i in range(self.num_components):
				if (diff > cluster[i][1]):
					diff = cluster[i][1]
					dim = i
			tuples.append((cluster[dim], dim))
		return tuples
		
	def modifyCoordinates(self, label, letter, factor):
		dim = self.getMoreImportantDimensionV2()[label][1]
		coordinates = self.__project(letter)
		coordinates[dim] += factor
		return self.__projectBack(coordinates)
		
	def testAlgo(self, factor):
		ls = []
		for aLetter in self.X_test:
			ls.append(self.modifyCoordinates(self.predict(np.array(aLetter).reshape(1, -1))[0], aLetter, factor))
		return ls
			
		
	def printLetter(self, letter):
		stroke.arrayToStroke(letter).plot()
    
    #first in blue
	def printLetters(self, letters):
		stroke.plot_list(map(lambda l: stroke.arrayToStroke(l), letters))
		
	def getEstimator(self):
		return self.estimator
	
	def getPrincipleComponents(self):
		return self.principleComponents
		
	def getParameterVariances(self):
		return self.parameterVariances
	
	def getMeanShape(self):
		return self.meanShape
		
if __name__ == '__main__':
    main()
