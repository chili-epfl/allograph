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
	
    dimension = 280
    images = buildImageCollection("/home/gdevecchi/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/gdevecchi/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress",'e',dimension)
    
    letters = []
    for key in images:
		for image in images[key]:
			kernel = np.ones((5, 5), "uint8")
			dilated = cv2.dilate(image,kernel,iterations=1)
			w, h = original_shape = tuple(image.shape)
			#~ plt.imshow(image,cmap="Greys")
			#~ plt.show()
			image_array = np.reshape(dilated, (w * h))
			#~ image_array = np.reshape(image, (w*h))
			letters.append(image_array)
			
    estimator = KMeans(n_clusters=computeNumberCentroids(letters), init='k-means++')
    #~ estimator = KMeans(n_clusters=5,init='k-means++')
    estimator.fit(letters) 
    
    for centroid in estimator.cluster_centers_:
		plt.imshow(np.reshape(centroid, (dimension,dimension)), cmap="Greys")
		plt.show()


def computeNumberCentroids(images):
    range_nb_clusters = range(2, 10)

    minSihouettes = 2.0
    minCluster = 1

    for n_clusters in range_nb_clusters:

        clusterer = KMeans(n_clusters=n_clusters, init='k-means++')

        cluster_labels = clusterer.fit_predict(images)

        silhouette_avg = silhouette_score(np.array(images), cluster_labels)

        if (silhouette_avg < minSihouettes):
            minSihouettes = silhouette_avg
            minCluster = n_clusters

    print "nb of centroids: "
    print minCluster
    return minCluster


def buildImageCollection(path1, path2, letter, dimension):
	strokes = {}
	images = {}
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
	
	for key in strokes:
		images[key] = []
		strokes[key] = stroke.childFromRobot(strokes[key])
		for aLetter in strokes[key]:
			images[key].append(aLetter.strokeToImage(dimension)) 
			
	return images
	
if __name__ == '__main__':
    main()

