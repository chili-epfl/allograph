#!/usr/bin/env python
# coding: utf-8

import LetterLearner as l1
import LetterLearnerV2 as l2
import LetterLearnerwoClusterization as oldL
import numpy as np
import matplotlib.pyplot as plt
import stroke

def main():
	newLetterLearner = l1.LetterLearner(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'a', 6, 10)
	oldLetterLearner = oldL.LetterLearnerwoClusterization(10, newLetterLearner.X_train)
	#~ l = l2.LetterLearnerV2(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'a', 6, 100)
	newLetterLearner.clusterize()
	clf = newLetterLearner.classify()
	newLetterLearner.performPCA()
	oldLetterLearner.performPCA()
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
	#~ for (le1, le2, le3, le4, le5, le6, le7, le8, le9) in zip(newLetterLearner.testAlgo(0.0), newLetterLearner.testAlgo(-1.0), newLetterLearner.testAlgo(-0.5), newLetterLearner.testAlgo(0.5), newLetterLearner.testAlgo(1.0), oldLetterLearner.testAlgo(xTest, -1.0), oldLetterLearner.testAlgo(xTest,-0.5), oldLetterLearner.testAlgo(xTest,0.5), oldLetterLearner.testAlgo(xTest,1.0)):
		
		#~ print "-1.0: ", stroke.euclidian_distance(stroke.arrayToStroke(le2), stroke.arrayToStroke(le6))
		#~ print "-0.5: ", stroke.euclidian_distance(stroke.arrayToStroke(le3), stroke.arrayToStroke(le7))
		#~ print "0.5: ", stroke.euclidian_distance(stroke.arrayToStroke(le4), stroke.arrayToStroke(le8))
		#~ print "1.0: ", stroke.euclidian_distance(stroke.arrayToStroke(le5), stroke.arrayToStroke(le9))
		
		#~ plt.subplot(2,2,1)
		#~ plt.title("original")
		#~ newLetterLearner.printLetter(le1)
		#~ plt.subplot(2,2,2)
		#~ plt.title("original vs new algo")
		#~ newLetterLearner.printLetters([le1, le2, le3, le4, le5])
		#~ plt.subplot(2,2,3)
		#~ plt.title("original vs old algo")
		#~ newLetterLearner.printLetters([le1, le6, le7, le8, le9])
		#~ plt.subplot(2,2,4)
		#~ plt.title(" original vs new algo vs old algo")
		#~ newLetterLearner.printLetters([le1, le5, le9])
		#~ plt.show()
		
	newLetterLearner.childrenNamePerCluster()



if __name__ == '__main__':
    main()
