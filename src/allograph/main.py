#!/usr/bin/env python
# coding: utf-8

import LetterLearner as l1
import LetterLearnerOnClusters as l2
import LetterLearnerwoClusterization as oldL
import numpy as np
import matplotlib.pyplot as plt
import stroke
 

def main():
	l = l1.LetterLearner(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'a', 6, 10)
	#~ oldLetterLearner = oldL.LetterLearnerwoClusterization(10, newLetterLearner.X_train)
	#~ l = l2.LetterLearnerOnClusters(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'a', 6, 100)
	#~ newLetterLearner.clusterize()
	#~ clf = newLetterLearner.classify()
	#~ newLetterLearner.performPCA()
	l.clusterize()
	l.classify()
	l.performPCA()
	for le1, le2, le3 in zip(l.testAlgoV2(0.0), l.testAlgoV2(0.5), l.testAlgoV3([0.5, 0.25])):
		l.printLetters([le1, le2, le3])
		plt.title("original vs transformation +0.5 vs +0.5,+0.25")
		plt.show()
	#~ oldLetterLearner.performPCA()
	#~ print ll.numShapesInDataset
	#~ print "///////////////////////////////////////////////////////////////PRINCIPLE COMPONENTS///////////////////////////////////////////////////////////////////"
	#~ print ll.getPrincipleComponents()
	#~ print l.getMoreImportantDimensionV2()
	#~ print l._projectClustersV2()
	#~ for aCentroid in zip(l.getEstimator().cluster_centers_, l.getEstimator().labels_):
		#~ #first in blue, second in red
		#~ li = [aCentroid[0]]
		#~ for i in np.linspace(-2.0, 2.0, 10):
			#~ li.append(l.modifyCoordinates(aCentroid[1], aCentroid[0], i))
		#~ l.printLetters(li)
	#~ xTest = newLetterLearner.X_test
	#~ a = []
	#~ b = []
	#~ newLetterLearner.infoDataSet(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"])
	"""///////////////////////////////////////////////////////////////////////////////////////TEST OLD VS NEW///////////////////////////////////////////////////////////////////////////////////////////////////////////////"""
	#~ for (le1, le2, le3, le4, le5, le6, le7, le8, le9) in zip(newLetterLearner.testAlgoV1(0.0), newLetterLearner.testAlgoV1(-2.0), newLetterLearner.testAlgoV1(-1.0), newLetterLearner.testAlgoV1(1.0), newLetterLearner.testAlgoV1(2.0),
	 #~ oldLetterLearner.testAlgo(xTest, -2.0), oldLetterLearner.testAlgo(xTest,-1.0), oldLetterLearner.testAlgo(xTest,1.0), oldLetterLearner.testAlgo(xTest,2.0)):
	#~ for (le5, le9) in zip(newLetterLearner.testAlgoV1(1.0), oldLetterLearner.testAlgo(xTest,1.0)):
		#~ a.append(stroke.euclidian_distance(stroke.arrayToStroke(le5), stroke.arrayToStroke(le9)))
		#~ b.append(le9)
		#~ print "-2.0: ", stroke.euclidian_distance(stroke.arrayToStroke(le2), stroke.arrayToStroke(le6))
		#~ print "-1.0: ", stroke.euclidian_distance(stroke.arrayToStroke(le3), stroke.arrayToStroke(le7))
		#~ print "1.0: ", stroke.euclidian_distance(stroke.arrayToStroke(le4), stroke.arrayToStroke(le8))
		#~ print "2.0: ", stroke.euclidian_distance(stroke.arrayToStroke(le5), stroke.arrayToStroke(le9))
		#~ if (stroke.euclidian_distance(stroke.arrayToStroke(le5), stroke.arrayToStroke(le9)) != 0):
			#~ plt.subplot(2,2,1)
			#~ plt.title("original")
			#~ newLetterLearner.printLetter(le1)
			#~ plt.subplot(2,2,2)
			#~ plt.title("new algo")
			#~ newLetterLearner.printLetter(le5)
			#~ plt.subplot(2,2,3)
			#~ plt.title("old algo")
			#~ newLetterLearner.printLetter(le9)
			#~ plt.subplot(2,2,4)
			#~ plt.title("new algo vs old algo")
			#~ newLetterLearner.printLetters([le5,le9])
			#~ plt.show()
	#~ print len(a)
	#~ plt.xlabel('id of letter in test set')
	#~ plt.ylabel('euclidian distance')
	#~ plt.plot(a)
	#~ plt.show()
	
	#~ xTest = newLetterLearner.X_test
	#~ for res in newLetterLearner.testAlgoV1(1.0):
		#~ print res[1]
		#~ plt.title(" original vs transformed")
		#~ newLetterLearner.printLetters([newLetterLearner.getEstimator().cluster_centers_[res[1]], res[0]])
		#~ plt.show()
		
	newLetterLearner.childrenNamePerCluster()
	
	#~ xTest = newLetterLearner.X_test
	#~ for (le1, le2, le3) in zip(newLetterLearner.testAlgoV2([0.0], 1), newLetterLearner.testAlgoV2([1.0, 0.3], 2), oldLetterLearner.testAlgo(xTest,1.0)):
		
		
		#~ plt.subplot(2,2,1)
		#~ plt.title("original")
		#~ newLetterLearner.printLetter(le1)
		#~ plt.subplot(2,2,2)
		#~ plt.title("new algo")
		#~ newLetterLearner.printLetter(le2)
		#~ plt.subplot(2,2,3)
		#~ plt.title("old algo")
		#~ newLetterLearner.printLetter(le3)
		#~ plt.subplot(2,2,4)
		#~ plt.title(" original vs new algo vs old algo")
		#~ newLetterLearner.printLetters([le1, le2, le3])
		#~ plt.show()

if __name__ == '__main__':
    main()
