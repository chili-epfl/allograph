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

	"""/////////////////////////////////////////////MANIPULATION////////////////////////////////////////////////////"""

	"""Get the children's strokes from the robot's"""
	for key in strokes:
		strokes[key] = stroke.childFromRobot(strokes[key])

	letters = []

	for key in strokes:
		print key
		for aStroke in strokes[key]:
			letters.append(stroke.strokeToArray(aStroke))
	#~ print len(letters)

	minBIC = sys.float_info.max
	minCluster = 1;
	for i in range(1,100):
		
		gmm = GMM(n_components=i)
		gmm.fit(letters)
		BIC = gmm.bic(np.array(letters))
		if (BIC < minBIC):
			minBIC = BIC;
			minCluster = i

	print minCluster
	
	gmm = GMM(n_components=minCluster)
	labels = gmm.fit_predict(letters)
	print labels
	



if __name__ == '__main__':
	main()

