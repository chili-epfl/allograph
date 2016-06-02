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


def main():
	ll = LetterLearner(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'a', 3)
	ll.clusterize()
	#~ clf = ll.classify()
	ll.performPCA()
	#~ print ll.numShapesInDataset
	#~ print "///////////////////////////////////////////////////////////////PRINCIPLE COMPONENTS///////////////////////////////////////////////////////////////////"
	#~ print ll.getPrincipleComponents()
	print ll.projectClustersV2()
	#~ print "///////////////////////////////////////////////////////////////MEAN SHAPE///////////////////////////////////////////////////////////////////"
	#~ print ll.getMeanShape()
	#~ print "///////////////////////////////////////////////////////////////PARAMETER VARIANCES///////////////////////////////////////////////////////////////////"
	#~ print ll.getParameterVariances()

class LetterLearner:
	strokes = {}
	letters = []
	letter = None
	estimator = None
	numShapesInDataset = 0
	numPointsInShapes = 70
	num_components = 10
	meanShape = 0
	principleComponents = None
	parameterVariances = None
	
	
	
	nbClusters = 0
	def __init__(self, folderNames, aLetter, clusters):
		self.nbClusters = clusters
		self.letter = aLetter
		self.strokes = self.builStrokeCollection(folderNames, aLetter)
		"""Get the children's strokes from the robot's"""
		for key in self.strokes:
			self.strokes[key] = stroke.childFromRobot(self.strokes[key])

		"""Build an array of all the strokes of all the children"""
		for key in self.strokes:
			for aStroke in self.strokes[key]:
				self.letters.append(stroke.strokeToArray(aStroke))
				self.numShapesInDataset = self.numShapesInDataset + 1
				
				
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
		
	def predict(self, clf, aStroke):
		return clf.predict(stroke.strokeToArray(aStroke).reshape(1,-1))
		
		
	def performPCA(self):
		dataMat = np.empty((self.numShapesInDataset, self.numPointsInShapes*2))
		i = 0
		for key in self.strokes:
			for aStroke in self.strokes[key]:
				dataMat[i] = stroke.strokeToArray(aStroke)
				i = i + 1 
				
		covarMat = np.cov(dataMat.T)
		eigVals, eigVecs = np.linalg.eig(covarMat)
		self.principleComponents = np.real(eigVecs[:, 0:self.num_components])
		self.parameterVariances = np.real(eigVals[0:self.num_components])
		self.meanShape = dataMat.mean(0).reshape((self.numPointsInShapes * 2, 1))
		
	def projectClusters(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components))
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			projected[i] = self.principleComponents.T.dot(aCentroid)
			i = i + 1
		return projected
		
	def projectClustersV2(self):
		projected = np.empty((len(self.estimator.cluster_centers_), self.num_components, 3))
		i = 0
		for aCentroid in self.estimator.cluster_centers_:
			filteredLetters = filter(lambda x: self.estimator.predict(np.array(x).reshape(1,-1)) == i, self.letters)
			projectedLetters = map(lambda letter: self.principleComponents.T.dot(letter), filteredLetters)
			
			tuples = []
			for j in range(self.num_components):
				mean = 0
				for letter in projectedLetters:
					mean = mean + letter[j]
				mean = mean/len(projectedLetters)
				tuples.append((mean - 2*(math.sqrt(self.parameterVariances[j])/math.sqrt(len(projectedLetters))), mean + 2*(math.sqrt(self.parameterVariances[j])/math.sqrt(len(projectedLetters)))))
				
			finalTuples = map(lambda tup: (tup[0][0], tup[1], tup[0][1]), zip(tuples, self.projectClusters()[i]))
			projected[i] = np.array(finalTuples)
			i = i + 1
		return projected
		
	def getPrincipleComponents(self):
		return self.principleComponents
		
	def getParameterVariances(self):
		return self.parameterVariances
	
	def getMeanShape(self):
		return self.meanShape
		
if __name__ == '__main__':
    main()
