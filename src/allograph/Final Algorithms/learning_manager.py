#!/usr/bin/env python

"""
learning allograph of letters from demonstration 
using different strategies and different metrics
"""

import logging; shapeLogger = logging.getLogger("shape_logger")
import os.path
from recordtype import recordtype
import glob
import numpy as np
import stroke
from allograph.stroke import Stroke
from ast import literal_eval


# global variables :
#-------------------

Shape = recordtype('Shape', [('path', None), ('shapeID', None), ('shapeType', None), ('shapeType_code', None), ('paramsToVary', None), ('paramValues', None)])

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"


# learning_manager class :
#-------------------------

class LearningManager():

	def __init__(self, generate_path, demo_path, robot_path, ref_path, mode = 'midway', all_children_paths = None):														 
		self.generate_path = generate_path
		self.demo_path = demo_path
		self.robot_path = robot_path
		self.ref_path = ref_path
		self.mode = mode
		self.all_children_paths = all_children_paths
		self.robot_data = read_data(self.robot_path,0)
		#/////////////////////////////ADDED///////////////////////////////////
		self.estimator = {}
		self.nbClusters = 6
		self.num_components = 10
		if (self.mode =='PCA_cluster'):
			if (self.all_children_paths) :
				for letter in alphabet:
					self.estimator[letter] = stroke.clusterize(self.getDataSet(letter), self.nbClusters)
        
			else:
				self.all_children_paths = self.robot_path
		#/////////////////////////////END ADDED///////////////////////////////////
		self.generated_letters = {}
		self.generate_letters()
		self.word_seen = set()
		self.current_word = ""
		self.refs = read_ref_data(self.ref_path,6) #6=line of the ref in dataset
		
		

	# the path to the ref is something like :
	#fileName = inspect.getsourcefile(ShapeModeler)
	#installDirectory = fileName.split('/lib')[0]
	#refDirectory =  installDirectory + '/share/shape_learning/letter_model_datasets/bad_letters'

	def word_to_learn(self, word):
		self.current_word = word
		self.word_seen.add(word)

	def generate_word(self, word):
		generated_word = []
		for letter in self.current_word:
			generated_word.append(self.generated_letters[letter])
		return generated_word

	def generate_letters(self):
		if (self.mode == 'last_state'):
			for letter in alphabet:
				aStroke = self.robot_data[letter][-1]
				self.generated_letters[letter] = aStroke
		if (self.mode == 'PCA_cluster'):
			for letter in alphabet:
				aStroke = self.robot_data[letter][-1]
				self.generated_letters[letter] = stroke.modifyCoordinatesRandom(aStroke, self.getDataSet(letter), self.estimator[letter], 0.2, self.num_components)
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)
     
		
	def respond_to_demonstration_word(self, demonstrations): #mutual_modeling will act here
		if (mode == 'midway'):
			for letter,stroke in demonstrations:
				learned_stroke = stroke.midway(stroke, self.generated_letters[letter])
				self.generated_letters[letter] = learned_stroke
				save_learned_allograph(self.robot_data, letter, learned_stroke)
				score = stroke.euclidian_distance(demo_stroke, self.refs[letter])
		if (mode == 'PCA_cluster') :
			for letter,aStroke in demonstrations:
				aStroke = stroke.modifyCoordinates(aStroke, self.getDataSet(letter), self.estimator[letter], 1.0, self.num_components)
				learned_stroke = stroke.midway(aStroke, self.generated_letters[letter])
				self.generated_letters[letter] = learned_stroke
				save_learned_allograph(self.robot_data, letter, learned_stroke)
				score = stroke.euclidian_distance(demo_stroke, self.refs[letter])
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)

	def respond_to_demonstration_letter(self, demonstration, letter, grade):
		demo_stroke = Stroke()
		demo_stroke.stroke_from_xxyy(np.reshape(demonstration,len(demonstration)))
		#demo_stroke.uniformize()
		demo_stroke.normalize_wrt_max()
		if mode == 'midway':
			learned_stroke = stroke.midway(demo_stroke, self.generated_letters[letter], grade)
			self.generated_letters[letter] = learned_stroke
			save_learned_allograph(self.robot_path, letter, learned_stroke)
			_,score = stroke.euclidian_distance(demo_stroke, self.refs[letter])
		if mode == 'PCA_cluster':
			demo_stroke = stroke.modifyCoordinates(demo_stroke, self.getDataSet(letter), self.estimator[letter], 1.0, self.num_components)
			learned_stroke = stroke.midway(demo_stroke, self.generated_letters[letter], grade)
			self.generated_letters[letter] = learned_stroke
			save_learned_allograph(self.robot_path, letter, learned_stroke)
			_,score = stroke.euclidian_distance(demo_stroke, self.refs[letter])
			 
		#if mode = 'sigNorm' (mixture of sigma-log-normal)
		#if mode = 'CNN' (1-D convolutionnal neural networks)
		return self.shape_message(letter),score


	def shape_message(self, letter):
		stroke = self.generated_letters[letter]
		path = np.concatenate((stroke.x, stroke.y))
		shape = Shape(path=path, shapeType=letter)
		return shape

	def shape_message_word(self):
		shapes = []
		for letter in self.current_word:
			shapes.append(self.shape_message(letter))
		return shapes

	def seen_before(self, word):
		return (word in self.word_seen)
      
	#/////////////////////////////ADDED///////////////////////////////////  
	"""Builds the collection of strokes from list of root folders."""	
	def buildStrokeCollectionOfAll(self, folderNames, letter):
		
		strokesPerChild = {}
		
		for folderName in folderNames:
			for root, dirs, files in os.walk(folderName):
				for name in dirs:
					strokesPerChild[name] = read_data(os.path.join(root,name),0)[letter]
		
		return strokesPerChild 
	
	def getDataSet(self, letter):
		strokesPerChild = self.buildStrokeCollectionOfAll(self.all_children_paths, letter)
		dataSet = []
		"""Get the children's strokes from the robot's."""
		for child in strokesPerChild:
			strokesPerChild[child] = stroke.childDemoFromRobotStroke(strokesPerChild[child])

		"""Builds an array of all the strokes of all the children and a list of all the children."""
		for child in strokesPerChild:
			for aStroke in strokesPerChild[child]:
				dataSet.append(stroke.strokeToArray(aStroke))
		return dataSet
      

# static functions :
#-------------------
def read_data(datasetDirectory, lines_to_jump):
    data_letters = {}
    datasets = glob.glob(datasetDirectory + '/*.dat')
    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]
        if name in alphabet:
            try:
                with open(dataset, 'r') as data:
                    i=0
                    for line in data.readlines():
                        i+=1
                        if i>lines_to_jump:
                            shape = literal_eval(('['+ line +']').replace('[,','['))
                            shape = np.reshape(shape, (-1, 1))
                            data_stroke = Stroke()
                            data_stroke.stroke_from_xxyy(np.reshape(shape,len(shape)))
                            data_stroke.uniformize()
                            data_stroke.downsampleShape(70)
                            try:
                                data_stroke.normalize_wrt_max()
                            except ValueError:
                                print name+" is empty"
                            data_letters.setdefault(name,[]).append(data_stroke)
            except IOError:
                raise RuntimeError("no reading permission for file"+dataset )
    return data_letters

def read_ref_data(refDirectory, line_of_ref):
    ref_letters = {}
    datasets = glob.glob(refDirectory + '/*.dat')
    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]
        if name in alphabet:
            try:
                with open(dataset, 'r') as data:
                    i=0
                    for line in data.readlines():
                        i+=1
                        if i==line_of_ref:
                            shape = literal_eval('['+line.replace(' ',', ')+']')
                            shape = np.reshape(shape, (-1, 1))
                            data_stroke = stroke.Stroke()
                            data_stroke.stroke_from_xxyy(np.reshape(shape,len(shape)))
                            data_stroke.uniformize()
                            data_stroke.normalize()
                            ref_letters[name] = data_stroke
                            break
            except IOError:
                raise RuntimeError("no reading permission for file"+dataset )

    return ref_letters


def save_learned_allograph(datasetDirectory, letter, stroke):
    dataset = datasetDirectory + '/' + letter + '.dat'
    shape_path = np.concatenate((np.array(stroke.x), np.array(stroke.y)))
    if not os.path.exists(dataset):
        raise RuntimeError("path to dataset"+dataset+"not found")
    try:
        with open(dataset, "a") as f:
            f.write(','.join(map(str,shape_path)))
            f.write('\n')
    except IOError:
        raise RuntimeError("noewriting permission for file"+dataset)
