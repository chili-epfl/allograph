#!/usr/bin/env python
# coding: utf-8


import stroke
import learning_manager as lm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn import svm
import math
from string import ascii_lowercase
import pandas


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
	principleValues = None
	X_train = []
	X_test = []
	children_train = []
	children_test = []
	children_strokes = None
	nbClusters = 0
	
	def __init__(self, folderNames, aLetter, clusters, components):
		self.nbClusters = clusters
		self.letter = aLetter
		self.strokes = self.buildStrokeCollection(folderNames, aLetter)
		self.num_components = components
		
		"""Get the children's strokes from the robot's."""
		for key in self.strokes:
			self.strokes[key] = stroke.childDemoFromRobotStroke(self.strokes[key])

		"""Builds an array of all the strokes of all the children and a list of all the children."""
		for key in self.strokes:
			for aStroke in self.strokes[key]:
				self.letters.append(stroke.strokeToArray(aStroke))
				self.children.append(key)
			
				
		"""splits the sets of letters in two sets: one to train, the other to test."""	
		self.X_train, self.X_test, self.children_train, self.children_test = train_test_split(np.array(self.letters,dtype=np.float64), np.array(self.children, dtype=np.str_), test_size=0.1)
		"""Saves the number of samples"""
		self.numShapesInDataset = len(self.X_train)
		
	"""Builds the collection of strokes from list of root folders."""	
	def buildStrokeCollection(self, folderNames, letter):
		
		strokes = {}
		
		for folderName in folderNames:
			for root, dirs, files in os.walk(folderName):
				for name in dirs:
					strokes[name] = lm.read_data(os.path.join(root,name),0)[letter]
		
		return strokes
		
	"""Prints the number of children per cluster"""
	def childrenNamePerCluster(self):
		su = [0]*self.nbClusters
		for s in range(0,999):
			i = 0
			for aCluster in self.estimator.cluster_centers_:
				j = 0
				lis = []
				for let in self.X_train:
					if (self.estimator.predict(np.array(let).reshape(1,-1)) == i):
						ok = True
						for elem in lis:
							if (self.children_train[j] == elem):
								ok = False
								#~ continue
						if (ok):
							lis.append(self.children_train[j])
					j += 1
				su[i] += len(lis) 
				i += 1
		for i in range(len(su)):
			su[i] /= float(1000.0)
		print su
		plt.plot(su)
		plt.title('Number of children per cluster')
		plt.xlabel('id of clusters')
		plt.ylabel('# children')
		plt.show()
				
				
	"""Display number of strokes per letter in the data set"""			
	def infoDataSet(self, folderNames):
		lis = {}	
		for l in ascii_lowercase:
			s = self.buildStrokeCollection(folderNames, l)
			print "Number of children", len(s)
			n = 0
			for key in s:
				for aStroke in s[key]:
					n += 1
			lis[l] = n
		df = pandas.DataFrame.from_dict(lis, orient='index')
		df.plot(kind='bar')
		plt.xlabel('letters')
		plt.ylabel('# strokes')
		plt.show()
			
			
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
		
	"""Perform PCA on a number of eigen vectors specified during the instanciation of the class"""
	def performPCA(self):
		"""Creating the matrix to project"""
		dataMat = np.array(self.X_train).reshape((self.numShapesInDataset, self.numPointsInShapes*2))
		
		"""Creating the covariance matrix"""
		covarMat = np.cov(dataMat.T)
		
		"""Generating the eigen vectors and eigen values"""
		eigVals, eigVecs = np.linalg.eig(covarMat)

		"""Taking the first num_components eigen vectors and values, and the center of the space."""
		self.principleComponents = np.real(eigVecs[:, 0:self.num_components])
		self.principleValues = np.real(eigVals[0:self.num_components])
		self.meanShape = dataMat.mean(0).reshape((self.numPointsInShapes * 2, 1))
		
		
	"""Helper function to project a letter on the eigen space"""
	def __project(self, letter):
		return np.dot(self.principleComponents.T, (letter.reshape(-1, 1) - self.meanShape)).reshape(self.num_components,)
	
	"""Helper function to project a letter back to the regular space"""
	def __projectBack(self, letter):
		return (self.meanShape + np.dot(self.principleComponents, letter).reshape(-1, 1)).reshape(self.numPointsInShapes*2, )
	
	"""Helper funciton to project all the centroids to the eigen space"""	
	def _projectCentroids(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components))
		i = 0
		"""loop over the centroids and project them"""
		for aCentroid in self.estimator.cluster_centers_:
			projected[i] = self.__project(aCentroid)
			i = i + 1
		return projected
	
	"""The following sections try to use the clusterization to be able to get the more important dimension in the eigenspace."""
	
	
	"""//////////////////////////////////////////////////////////////////////////////First version////////////////////////////////////////////////////////////////////////////////////////////////////"""
	"""This first version use the confidence interval for each clusters, for each cluster it will compute the confidence interval and select the dimension of the eigenspace that has the smallest one."""
	
	"""Construct an array of confidence interval for each dimension of each cluster"""
	def _projectClustersV1(self):
		"""Array of confidence interval for each cluster and each components"""
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components, 3))
		i = 0
		"""loop over the centroids"""
		for aCentroid in self.estimator.cluster_centers_:
			"""take the letters in the cluster"""
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.X_train)
			"""project the previous list on the eigen space"""
			projectedLetters = map(lambda letter: self.__project(letter), filteredLetters)
			
			tuples = []
			"""loop over the eigen vectors"""
			for j in range(self.num_components):
				mean = 0
				for letter in projectedLetters:
					mean = mean + letter[j]
				"""Compute the mean"""
				mean = mean/len(projectedLetters)
				"""compute the confidence interval for this dimension and add it to 'tuples' """
				tuples.append((mean - 2*(math.sqrt(self.parameterVariances[j])/math.sqrt(len(projectedLetters))), mean + 2*(math.sqrt(self.parameterVariances[j])/math.sqrt(len(projectedLetters)))))
				
			"""Construct a list of confidence interval and coordinates of the centroid for each dimension"""
			finalTuples = map(lambda tup: (tup[0][0], tup[1], tup[0][1]), zip(tuples, self._projectCentroids()[i]))
			
			projected[i] = np.array(finalTuples)
			i = i + 1
		return projected
	
	"""Select the smallest confidence interval dimension for each cluster"""
	def _getMoreImportantDimensionV1(self):
		tuples = []
		for cluster in self._projectClustersV1():
			dim = -1
			diff = np.Infinity
			for i in range(self.num_components):
				if (diff > (cluster[i][2]-cluster[i][0])):
					diff = cluster[i][2]-cluster[i][0]
					dim = i
			tuples.append(dim)
		return tuples
		
	"""Projects onto the eigenspace modify the most important coordinate and project back to the regular space"""
	def _modifyCoordinatesV1(self, label, letter, factor):
		dim = self._getMoreImportantDimensionV1()[label]
		coordinates = self.__project(letter)
		coordinates[dim] += factor
		return self.__projectBack(coordinates)
	
	"""modify all the letters of the test set and return them"""
	def testAlgoV1(self, factor):
		ls = []
		for aLetter in self.X_test:
			label = self.predict(np.array(aLetter).reshape(1, -1))[0]
			ls.append(self._modifyCoordinatesV1(label, aLetter, factor))
		return ls
		
	"""//////////////////////////////////////////////////////////////////////////////Second Version////////////////////////////////////////////////////////////////////////////////////////////////////"""
	"""This version computes for each dimension of the eigenspace the normalized variance for each cluster and selects the smallest one for each cluster"""
	
	"""Computes for each cluster the normalized variance on each dimension"""
	def _projectClustersV2(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components, 2))
		i = 0
		"""loop over the centroids"""
		for aCentroid in self.estimator.cluster_centers_:
			"""Take only the letters in the cluster"""
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.X_train)
			"""Project the letters on the eigenspace"""
			projectedLetters = map(lambda letter: self.__project(letter), filteredLetters)
			varNormalized = []
			"""loop over the eigenvectors"""
			for j in range(self.num_components):
				"""Compute the normalized variance"""
				varNormalized.append(np.var(map(lambda x: x[j], projectedLetters))/self.principleValues[j])
			"""create final tuple with the normalized variance and the coordinate of the centroid"""
			finalTuples = zip(varNormalized, self._projectCentroids()[i])
			projected[i] = np.array(finalTuples)
			i = i + 1
		return projected

	"""Select the smallest variance dimension of the eigenspace"""
	def _getMoreImportantDimensionV2(self):
		tuples = []
		for cluster in self._projectClustersV2():
			dim = -1
			diff = np.Infinity
			for i in range(self.num_components):
				if (diff > cluster[i][1]):
					diff = cluster[i][1]
					dim = i
			tuples.append(dim)
			
		return tuples
		
	"""Projects onto the eigenspace modify the most important coordinate and project back to the regular space"""
	def _modifyCoordinatesV2(self, label, letter, factor):
		dim = self._getMoreImportantDimensionV2()[label]
		coordinates = self.__project(letter)
		coordinates[dim] += factor
		return self.__projectBack(coordinates)
		
	"""modify all the letters of the test set and return them"""
	def testAlgoV2(self, factor):
		ls = []
		for aLetter in self.X_test:
			label = self.predict(np.array(aLetter).reshape(1, -1))[0]
			ls.append(self._modifyCoordinatesV2(label, aLetter, factor))
		return ls
	
	"""//////////////////////////////////////////////////////////////////////////////Third Version////////////////////////////////////////////////////////////////////////////////////////////////////"""
	"""As in second version but gives n most important dimensions"""
	
	"""Computes for each cluster the normalized variance on each dimension"""
	def _projectClustersV3(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components, 2))
		i = 0
		"""loop over the centroids"""
		for aCentroid in self.estimator.cluster_centers_:
			"""Take only the letters in the cluster"""
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.X_train)
			"""Project the letters on the eigenspace"""
			projectedLetters = map(lambda letter: self.__project(letter), filteredLetters)
			varNormalized = []
			"""loop over the eigenvectors"""
			for j in range(self.num_components):
				"""Compute the normalized variance"""
				varNormalized.append((np.var(map(lambda x: x[j], projectedLetters))/self.principleValues[j], j))
			"""create final tuple with the normalized variance and the coordinate of the centroid"""
			projected[i] = np.array(varNormalized)
			i = i + 1
		return projected

	"""Select the n smallest variance dimension of the eigenspace"""
	def _getMoreImportantDimensionV3(self, n):
		tuples = []
		for cluster in self._projectClustersV3():
			def getKey(elem):
				return elem[0]
			lis = sorted(cluster, key=getKey)
			tuples.append(map(lambda elem: int(elem[1]), lis[:n]))
			
		return tuples

	"""Modify coordinates on n most important eigenvectors"""
	def _modifyCoordinatesV3(self, label, letter, factors):
		dims = self._getMoreImportantDimensionV3(len(factors))[label]
		coordinates = self.__project(letter)
		i = 0
		for dim in dims:
			coordinates[dim] += factors[i]
			i += 1
			
		return self.__projectBack(coordinates)
	
	"""modify all the letters of the test set and return them"""
	def testAlgoV3(self, factors):
		ls = []
		for aLetter in self.X_test:
			ls.append(self._modifyCoordinatesV3(self.predict(np.array(aLetter).reshape(1, -1))[0], aLetter, factors))
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
