#!/usr/bin/env python
# coding: utf-8

import learning_manager as lm
import stroke
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM

def main():
	"""/////////////////////////////////////////////INITIALIZATION/////////////////////////////////////"""

	""" Create dictionnary : for a child -> strokes of the robot from the .dat files a the letter a"""
	strokes = buildStrokeCollection("/home/gdevecchi/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/gdevecchi/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress",'a')

	"""/////////////////////////////////////////////MANIPULATION////////////////////////////////////////////////////"""

	"""Get the children's strokes from the robot's"""
	for key in strokes:
		strokes[key] = stroke.childFromRobot(strokes[key])

	letters = []

	for key in strokes:
		print key
		for aStroke in strokes[key]:
			letters.append(stroke.strokeToArray(aStroke))

	minBIC = sys.float_info.max
	minCluster = 1;
	for i in range(1,100):
		
		gmm = DPGMM(n_components=i)
		gmm.fit(letters)
		BIC = gmm.bic(np.array(letters))
		if (BIC < minBIC):
			minBIC = BIC;
			minCluster = i

	print minCluster
	
	gmm = DPGMM(n_components=minCluster)
	labels = gmm.fit_predict(letters)
	print labels
	

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

if __name__ == '__main__':
	main()

