#!/usr/bin/env python
# coding: utf-8

import LetterLearner as ll


def main():
	l = ll.LetterLearner(["/home/guillaume/Documents/projet_chili/cowriter_logs/Normandie/robot_progress","/home/guillaume/Documents/projet_chili/cowriter_logs/EIntGen/robot_progress"],'b', 6, 5)
	l.clusterize()
	#~ clf = ll.classify()
	l.performPCA()
	#~ print ll.numShapesInDataset
	#~ print "///////////////////////////////////////////////////////////////PRINCIPLE COMPONENTS///////////////////////////////////////////////////////////////////"
	#~ print ll.getPrincipleComponents()
	#~ print l.getMoreImportantDimensionV2()
	print l._projectClustersV2()
	#~ print "///////////////////////////////////////////////////////////////MEAN SHAPE///////////////////////////////////////////////////////////////////"
	#~ print l.getMeanShape()
	#~ print "///////////////////////////////////////////////////////////////PARAMETER VARIANCES///////////////////////////////////////////////////////////////////"
	#~ print l.getParameterVariances()



if __name__ == '__main__':
    main()
