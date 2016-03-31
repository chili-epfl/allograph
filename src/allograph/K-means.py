#!/usr/bin/env python
# coding: utf-8

import stroke
import learning_manager as lm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def main():
    """/////////////////////////////////////////////INITIALIZATION/////////////////////////////////////"""

    """ Create dictionnary : for a child -> strokes of the robot from the .dat files a the letter a"""
    strokes = buildStrokeCollection("/home/gdevecchi/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/gdevecchi/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress",'a')
    

    # ~ """Initialize Object KMeans that is goind to be used to find centroids on children's tries"""
    # ~ estimator = KMeans(init='k-means++')

    """/////////////////////////////////////////////MANIPULATION////////////////////////////////////////////////////"""

    """Get the children's strokes from the robot's"""
    for key in strokes:
	    strokes[key] = stroke.childFromRobot(strokes[key])

    """Build an array of all the strokes of all the children"""
    letters = []
    #~ print "x"
    #~ print strokes["WilliamRin"][0].get_x()
    #~ print "y"
    #~ print strokes["WilliamRin"][0].get_y()
    #~ np.set_printoptions(threshold='nan')
    #~ plt.imshow(strokes["WilliamRin"][0].strokeToImage(100))
    #~ plt.show()
    #~ plt.imshow(strokes["WilliamRin"][1].strokeToImage(100))
    #~ plt.show()
    #~ plt.imshow(strokes["WilliamRin"][2].strokeToImage(100))
    #~ plt.show()
    #~ plt.imshow(strokes["WilliamRin"][3].strokeToImage(100))
    #~ plt.show()
    #~ print strokes["WilliamRin"][1].strokeToImage(15)
    #~ print strokes["WilliamRin"][2].strokeToImage(15)
    #~ print strokes["WilliamRin"][3].strokeToImage(15)
	
    print len(letters)
    
    for key in strokes:
	for aStroke in strokes[key]:
	    letters.append(stroke.strokeToArray(aStroke))
			
    
    
    estimator = KMeans(n_clusters=computeNumberCentroids(letters), init='k-means++')
    # ~ estimator = KMeans(n_clusters=8,init='k-means++')

    """Compute KMean of all the letters a with 8 centroids (default)"""
    estimator.fit(letters)

    centroidStrokes = []

    """create strokes from centroids to be able to plot them easily"""
    for centroid in estimator.cluster_centers_:
        newStroke = stroke.Stroke()
        newStroke.stroke_from_xxyy(centroid)
        newStroke.downsampleShape(70)
        newStroke.uniformize()
        centroidStrokes.append(newStroke)

    """Plot the centroids"""
    for aCentroid in centroidStrokes:
		#~ aCentroid.plot()
		#~ plt.imshow(aCentroid.strokeToImage(100), cmap="Greys")
		#~ plt.show()
		kernel = np.ones((30, 30), "uint8")
		dilated = cv2.dilate(aCentroid.strokeToImage(3000),kernel,iterations=1)
		plt.imshow(dilated, cmap="Greys")
		plt.show()

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
    range_nb_clusters = range(2, 10)

    minSihouettes = 2.0
    minCluster = 1

    for n_clusters in range_nb_clusters:

        clusterer = KMeans(n_clusters=n_clusters, init='k-means++')

        cluster_labels = clusterer.fit_predict(letters)

        silhouette_avg = silhouette_score(np.array(letters), cluster_labels)

        if (silhouette_avg < minSihouettes):
            minSihouettes = silhouette_avg
            minCluster = n_clusters

    print "nb of centroids: "
    print minCluster
    return minCluster


if __name__ == '__main__':
    main()
