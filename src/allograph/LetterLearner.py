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
	estimator = ll.clusterize()
	clf = ll.classify(estimator)

class LetterLearner:
	strokes = {}
	letters = []
	letter = None
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
			
	def builStrokeCollection(self, folderNames, letter):
		strokes = {}
		
		for folderName in folderNames:
			for root, dirs, files in os.walk(folderName):
				for name in dirs:
					strokes[name] = lm.read_data(os.path.join(root,name),0)[letter]
				
		return strokes
		
	def clusterize(self):
		estimator = KMeans(n_clusters=self.nbClusters,init='k-means++')
		estimator.fit(self.letters)
		return estimator
		
	def classify(self,estimator):
		X_train, X_test, y_train, y_test = train_test_split(np.array(self.letters,dtype=np.float64),estimator.labels_)
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
		

if __name__ == '__main__':
    main()
