#!/usr/bin/env python
# coding: utf-8

import stroke
import learning_manager as lm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def main():
    """/////////////////////////////////////////////INITIALIZATION/////////////////////////////////////"""

    """ Create dictionnary : for a child -> strokes of the robot from the .dat files a the letter a"""
    strokes = {}
    strokes["Adele"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_adele", 0)['a']
    strokes["Alexandre"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_alexandre", 0)['a']
    strokes["Enzo"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_enzo", 0)['a']
    strokes["Jonathan"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_jonathan", 0)['a']
    strokes["Matenzo"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_matenzo", 0)['a']
    strokes["Mona"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_mona", 0)['a']
    strokes["Nathan"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_nathan", 0)['a']
    strokes["Valentine"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/Normandie/robot_progress/with_valentine", 0)['a']
    strokes["Avery"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_avery", 0)['a']
    strokes["Dan"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_dan", 0)['a']
    strokes["DanielEge"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_daniel_ege", 0)['a']
    strokes["Gaia"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_gaia", 0)['a']
    strokes["Ines"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_ines", 0)['a']
    strokes["JacquelineNadine"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_jacqueline_nadine",
                 0)['a']
    strokes["Jake"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_jake", 0)['a']
    strokes["LaithKayra"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_laith_kayra", 0)['a']
    strokes["Lamonie"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_lamonie", 0)['a']
    strokes["LilaLudovica"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_lila_ludovica", 0)[
        'a']
    strokes["loulwaAnais"] = \
    lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_loulwa_anais", 0)[
        'a']
    strokes["Markus"] = \
		lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_markus", 0)['a']
    strokes["OsborneAmelia"] = lm.read_data(
		"/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_osborne_amelia_enzoV", 0)['a']
    strokes["Oscar"] = \
	lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_oscar", 0)['a']
    strokes["WilliamRin"] = \
	lm.read_data("/home/guillaume/Documents/Projet CHILI/cowriter_logs/EIntGen/robot_progress/with_william_rin", 0)['a']

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
    print strokes["WilliamRin"][0].strokeToImage(10)
    print strokes["WilliamRin"][1].strokeToImage(10)
    print strokes["WilliamRin"][2].strokeToImage(10)
    print strokes["WilliamRin"][3].strokeToImage(10)
	
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
    #~ for aCentroid in centroidStrokes:
        #~ aCentroid.plot()


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
