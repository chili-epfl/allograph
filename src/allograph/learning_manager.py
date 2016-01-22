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
from allograph import stroke
from allograph.stroke import Stroke
from ast import literal_eval

# global variables :
#-------------------

Shape = recordtype('Shape', [('path', None), ('shapeID', None), ('shapeType', None), ('shapeType_code', None), ('paramsToVary', None), ('paramValues', None)])

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"


# learning_manager class :
#-------------------------

class LearningManager():

    def __init__(self, generate_path, demo_path, robot_path, ref_path):
        self.generate_path = generate_path
        self.demo_path = demo_path
        self.robot_path = robot_path
        self.ref_path = ref_path
        self.robot_data = read_data(self.robot_path,0)
        self.generated_letters = {}
        self.generate_letters('last_state')
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

    def generate_letters(self, mode='last_state'):
        if mode == 'last_state':
            for letter in alphabet:
                stroke = self.robot_data[letter][-1]
                self.generated_letters[letter] = stroke
        #if mode = 'PCA' 
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)

    def respond_to_demonstration_word(self, demonstrations, mode='midway'): #mutual_modeling will act here
        if mode == 'midway':
            for letter,stroke in demonstrations:
                learned_stroke = stroke.midway(stroke, self.generated_letters[letter])

                self.generated_letters[letter] = learned_stroke
                save_learned_allograph(self.robot_data, letter, learned_stroke)
                score = stroke.euclidian_distance(demo_stroke, self.refs[letter])
        #if mode = 'PCA' 
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)

    def respond_to_demonstration_letter(self, demonstration, letter, grade, mode='midway'):
        demo_stroke = Stroke()
        demo_stroke.stroke_from_xxyy(np.reshape(demonstration,len(demonstration)))
        #demo_stroke.uniformize()
        demo_stroke.normalize_wrt_max()
        if mode == 'midway':
            learned_stroke = stroke.midway(demo_stroke, self.generated_letters[letter], grade)
            self.generated_letters[letter] = learned_stroke
            save_learned_allograph(self.robot_path, letter, learned_stroke)
            score = stroke.euclidian_distance(demo_stroke, self.refs[letter])
        #if mode = 'PCA' 
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)
        return self.shape_message(letter),score


    def shape_message(self, letter):
        stroke = self.generated_letters[letter]
        path = np.concatenate((stroke.x, stroke.y))
        shape = Shape(path=path, shapeType=letter)

    def shape_message_word(self):
        shapes = []
        for letter in self.current_word:
            shapes.append(self.shape_message(letter))
        return shapes

    def seen_before(self, word):
        return (word in self.word_seen)

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
                            #data_stroke.uniformize()
                            data_stroke.normalize_wrt_max()
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
