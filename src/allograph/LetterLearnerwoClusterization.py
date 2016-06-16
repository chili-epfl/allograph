#!/usr/bin/env python
# coding: utf-8


import stroke
import learning_manager as lm
import os
import numpy as np

"""Simulates Deanna Hood's algorithm"""
class LetterLearnerwoClusterization:
	strokes = {}
	letters = []
	letter = None
	meanShape = 0
	principleComponents = None
	parameterVariances = None
	principleValues = None
	num_components = 0
	numShapesInDataset = 0
	numPointsInShapes = 70
	dataSet = None
	
	
	def __init__(self, components, data):
		self.dataSet = data
		self.num_components = components
		self.numShapesInDataset = len(self.dataSet)
		
	"""Performs PCA on the entire data set"""
	def performPCA(self):
		dataMat = np.array(self.dataSet).reshape((self.numShapesInDataset, self.numPointsInShapes*2))
		covarMat = np.cov(dataMat.T)
		eigVals, eigVecs = np.linalg.eig(covarMat)
		self.principleComponents = np.real(eigVecs[:, 0:self.num_components])
		self.principleValues = np.real(eigVals[0:self.num_components])
		self.parameterVariances = np.real(eigVals[0:self.num_components])
		self.meanShape = dataMat.mean(0).reshape((self.numPointsInShapes * 2, 1))

	"""Project a letter onto the eigenspace"""
	def __project(self, letter):
		return np.dot(self.principleComponents.T, (letter.reshape(-1, 1) - self.meanShape)).reshape(self.num_components,)
	
	"""Projects a letter back onto the regular space"""
	def __projectBack(self, letter):
		return (self.meanShape + np.dot(self.principleComponents, letter).reshape(-1, 1)).reshape(self.numPointsInShapes*2, )

	"""Modifies the coordinate on the first eigenvector (most variance)"""
	def modifyCoordinates(self, letter, factor):
		coordinates = self.__project(letter)
		coordinates[0] += factor
		return self.__projectBack(coordinates)
	
	"""Tests the algorithm on the data set"""
	def testAlgo(self, X_test, factor):
		ls = []
		for aLetter in X_test:
			ls.append(self.modifyCoordinates(aLetter, factor))
		return ls
