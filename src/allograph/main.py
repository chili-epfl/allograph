#!/usr/bin/env python
# coding: utf-8

import LetterLearner as l1
import LetterLearnerV2 as l2
import numpy as np

def main():
	l = l1.LetterLearner(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'e', 6, 10)
	#~ l = l2.LetterLearnerV2(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'a', 6, 100)
	l.clusterize()
	#~ clf = ll.classify()
	l.performPCA()
	#~ print ll.numShapesInDataset
	#~ print "///////////////////////////////////////////////////////////////PRINCIPLE COMPONENTS///////////////////////////////////////////////////////////////////"
	#~ print ll.getPrincipleComponents()
	#~ print l.getMoreImportantDimensionV2()
	#~ print l._projectClustersV2()
	for aCentroid in zip(l.getEstimator().cluster_centers_, l.getEstimator().labels_):
		#first in blue, second in red
		li = [aCentroid[0]]
		for i in np.linspace(-2.0, 2.0, 10):
			li.append(l.modifyCoordinates(aCentroid[1], aCentroid[0], i))
		l.printLetters(li)
	
	#~ print "///////////////////////////////////////////////////////////////MEAN SHAPE///////////////////////////////////////////////////////////////////"
	#~ print l.getMeanShape()
	#~ print "///////////////////////////////////////////////////////////////PARAMETER VARIANCES///////////////////////////////////////////////////////////////////"
	#~ print l.getParameterVariances()



if __name__ == '__main__':
    main()
