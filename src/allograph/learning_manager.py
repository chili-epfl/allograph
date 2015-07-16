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

    def __init__(self, generate_path, demo_path, robot_path):
        self.generate_path = generate_path
        self.demo_path = demo_path
        self.robot_path = robot_path
        self.robot_data = read_data(self.robot_path,0)
        self.generated_letters = {}
        self.generate_letters('last_state')
        self.word_seen = set()
        self.current_word = ""

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

    def respond_to_demonstration(self, demonstration, mode='midway'): #mutual_modeling will act here
        if mode == 'midway':
            for letter,stroke in demonstration:
                learned_stroke = stroke.midway(stroke, self.generated_letters[letter])
                self.generated_letters[letter] = learned_stroke
                save_learned_allograph(self.robot_data, letter, learned_stroke)
        #if mode = 'PCA' 
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)

    def shape_message(self, letter):
        stroke = generated_letters[letter]
        path = np.concatenate(stroke.x, stroke.y)
        shape = Shape(path=path, shapeType=letter)

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
                            data_stroke.uniformize()
                            data_stroke.normalize()
                            data_letters.setdefault(name,[]).append(data_stroke)
            except IOError:
                raise RuntimeError("no reading permission for file"+dataset )
    return data_letters

def save_learned_allograph(datasetDirectory, letter, stroke):
    dataset = datasetDirectory + '/' + letter + '.dat'
    if not os.path.exists(dataset):
        raise RuntimeError("path to dataset"+dataset+"not found")
    try:
        with open(dataset, "a") as f:
            f.write(','.join(map(str,stroke)))
            f.write('\n')
    except IOError:
        raise RuntimeError("no writing permission for file"+dataset)
