#!/usr/bin/env python
# coding: utf-8

import stroke
import learning_manager as lm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import cross_validation 
from sklearn import svm
from string import ascii_lowercase
from sklearn.decomposition import PCA
import math


def main():
	meanSum = 0
	lis = []
	#~ meanmean = [0]*len(ascii_lowercase)
	#~ for i in range(0, 99): 
	for j in range(len(ascii_lowercase)):
		"""/////////////////////////////////////////////INITIALIZATION/////////////////////////////////////"""

		""" Create dictionnary : for a child -> strokes of the robot from the .dat files a the letter a"""
		strokes = buildStrokeCollection("/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress",ascii_lowercase[j])
   
	
		"""/////////////////////////////////////////////MANIPULATION////////////////////////////////////////////////////"""

		"""Get the children's strokes from the robot's"""
		for key in strokes:
			strokes[key] = stroke.childFromRobot(strokes[key])

		"""Build an array of all the strokes of all the children"""
		letters = []
		X_train = []
		X_test = []
		
		for key in strokes:
			for aStroke in strokes[key]:
				letters.append(stroke.strokeToArray(aStroke))
		allButOne = []
		start = True
		y_test_labels = []
		y_train_labels = []
		i = 0
		for key in strokes:
			if (start):
				start = False
				
				for aStroke in strokes[key]:
					X_test.append(stroke.strokeToArray(aStroke))
					y_test_labels.append(i)
					i += 1
			else:		
				for aStroke in strokes[key]:
					allButOne.append(stroke.strokeToArray(aStroke))
					y_train_labels.append(i)
					i = i+1
    
		n_clusters = 6
			#~ n_clusters = computeNumberCentroids(letters)
		
		estimator = KMeans(n_clusters=n_clusters,init='k-means++')
			#~ pca = PCA(n_components=n_clusters).fit(letters)
			#~ estimator = KMeans(n_clusters=n_clusters,init=pca.components_,n_init=1)

		"""Compute KMean of all the letters a with 8 centroids (default)"""
		estimator.fit(letters)
    
		"""////////////////////////////////////////////////CLASSIFY//////////////////////////////////////////////////////"""
    
		#~ X_train, X_test, y_train, y_test = cross_validation.train_test_split(np.array(letters,dtype=np.float64),estimator.labels_, test_size=0.1)
		clf = svm.SVC()
		#~ clf.fit(X_train, y_train) 
		#~ cv = cross_validation.ShuffleSplit()
		scores = cross_validation.cross_val_score(clf, np.array(letters,dtype=np.float64),estimator.labels_,cv=10)
		#~ print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		lis.append(scores.mean())
	plt.plot(lis)
	plt.title('the mean accuracy of each letter of the alphabet from a test set')
	plt.xlabel('id of letter')
	plt.ylabel('mean accuracy')
	plt.show()
	print lis
		
		#~ y_test = []
		#~ y_train = []
		#~ for i in y_test_labels:
			#~ y_test.append(letters[i])
			
		#~ for i in y_train_labels:
			#~ y_train.append(letters[i])
			
		
		#~ labelsPredicted =  clf.predict(X_test)
   
		#~ diff = [predicted-real for predicted,real in zip(labelsPredicted, y_test)]
    
		#~ count = 0
		#~ for d in diff:
			#~ if (d == 0):
				#~ count = count + 1	
			
		#~ print (float(count)/len(diff))*100
		#~ meanmean[j] += accuracy
			#~ lis.append(accuracy)
			#~ meanSum += accuracy
	#~ for i in range(0, 99):
		#~ meanmean[j] /= float(100.0)
	#~ plt.plot(meanmean)
	#~ plt.show()
	#~ print meanmean
		
	
    
    

	"""////////////////////////////////////////////PRINT CENTROIDS///////////////////////////////////////////////////"""
		#~ centroidStrokes = []

		#~ """create strokes from centroids to be able to plot them easily"""
		#~ for centroid in estimator.cluster_centers_:
			#~ centroidStrokes.append(arrayToStroke(centroid))
        
    
    
		#~ """Plot the centroids"""
		#~ i = 0
		#~ for aCentroid in centroidStrokes:
			#~ n = 0
			#~ for j, value in enumerate(letters):
				#~ if (estimator.labels_[j] == i):
					#~ n += 1
					#~ s = arrayToStroke(value)
					#~ stroke.align(aCentroid, s)
					#~ print i
					#~ print stroke.euclidian_distance(aCentroid, s)[0]
					#~ aCentroid.plot_compare(s)
			#~ print i 
			#~ print n
			#~ i = i+1
			#~ aCentroid.plot()
    
	"""////////////////////////////////////////////CHARACTERIZATION///////////////////////////////////////////////////"""

	

def arrayToStroke(array):
    newStroke = stroke.Stroke()
    newStroke.stroke_from_xxyy(array)
    newStroke.downsampleShape(70)
    newStroke.uniformize()
    return newStroke
    
def buildStrokeCollection(path1, path2, letter):
	strokes = {}
	strokes["Adele"] = lm.read_data(path1 + "/with_adele", 0)[letter]
	strokes["Alexandre"] = lm.read_data(path1+"/with_alexandre", 0)[letter]
	strokes["Enzo"] = lm.read_data(path1+"/with_jonathan", 0)[letter]
	strokes["Matenzo"] = lm.read_data(path1+"/with_matenzo", 0)[letter]
	strokes["Mona"] = lm.read_data(path1+"/with_mona", 0)[letter]
	strokes["Nathan"] = lm.read_data(path1+"/with_nathan", 0)[letter]
	strokes["Valentine"] = lm.read_data(path1+"/with_valentine", 0)[letter]
	strokes["Avery"] = lm.read_data(path2+"/with_avery", 0)[letter]
	strokes["Dan"] = lm.read_data(path2+"/with_dan", 0)[letter]
	strokes["DanielEge"] = lm.read_data(path2+"/with_daniel_ege", 0)[letter]
	strokes["Gaia"] = lm.read_data(path2+"/with_gaia", 0)[letter]
	strokes["Ines"] = lm.read_data(path2+"/with_ines", 0)[letter]
	strokes["JacquelineNadine"] = lm.read_data(path2+"/with_jacqueline_nadine", 0)[letter]
	strokes["Jake"] = lm.read_data(path2+"/with_jake", 0)[letter]
	strokes["LaithKayra"] = lm.read_data(path2+"/with_laith_kayra", 0)[letter]
	strokes["Lamonie"] = lm.read_data(path2+"/with_lamonie", 0)[letter]
	strokes["LilaLudovica"] = lm.read_data(path2+"/with_lila_ludovica", 0)[letter]
	strokes["loulwaAnais"] = lm.read_data(path2+"/with_loulwa_anais", 0)[letter]
	strokes["Markus"] = lm.read_data(path2+"/with_markus", 0)[letter]
	strokes["OsborneAmelia"] = lm.read_data(path2+"/with_osborne_amelia_enzoV", 0)[letter]
	strokes["Oscar"] = lm.read_data(path2+"/with_oscar", 0)[letter]
	strokes["WilliamRin"] = lm.read_data(path2+"/with_william_rin", 0)[letter]
	return strokes
	

def computeNumberCentroids(letters):
    range_nb_clusters = range(2, len(letters)-1)

    minSihouettes = 2.0
    minCluster = 1

    for n_clusters in range_nb_clusters:

        clusterer = KMeans(n_clusters=n_clusters, init='k-means++')

        cluster_labels = clusterer.fit_predict(letters)

        silhouette_avg = silhouette_score(np.array(letters), cluster_labels)

        if (silhouette_avg < minSihouettes):
            minSihouettes = silhouette_avg
            minCluster = n_clusters

    print "nb of centroids:"
    print minCluster
    return minCluster


if __name__ == '__main__':
    main()
