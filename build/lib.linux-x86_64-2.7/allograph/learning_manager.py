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

    def __init__(self, child_path, ref_path, mode, mode_param):

        self.child_path = child_path
        self.ref_path = ref_path

        self.robot_data = read_data(self.child_path, 0)
        self.generated_letters = {}
        self.generate_letters('last_state')
        self.current_word = ""
        self.refs = read_ref_data(self.ref_path, 6) #6=line of the ref in dataset
        self.mode = mode
        self.mode_param = mode_param

    def word_to_learn(self, word):
        self.current_word = word

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

    def respond_to_demonstration_letter(self, demonstration, letter, mode='midway', mode_param=0.5):
        demo_stroke = Stroke()
        demo_stroke.stroke_from_xxyy(np.reshape(demonstration,len(demonstration)))
        demo_stroke.normalize_wrt_max()

        if mode == 'midway':
            learned_stroke = stroke.midway(demo_stroke, self.generated_letters[letter], mode_param)
            self.generated_letters[letter] = learned_stroke
            save_learned_allograph(self.child_path, letter, learned_stroke)
            _,score = stroke.euclidian_distance(demo_stroke, self.refs[letter])

        if mode == 'simple':
            learned_stroke = stroke.weigthedSum(demo_stroke, self.generated_letters[letter], mode_param)
            self.generated_letters[letter] = learned_stroke
            save_learned_allograph(self.child_path, letter, learned_stroke)
            _,score = stroke.euclidian_distance(demo_stroke, self.refs[letter])

        return self.shape_message(letter),score


    def shape_message(self, letter):
        stroke = self.generated_letters[letter]
        path = np.concatenate((stroke.x, stroke.y))
        shape = Shape(path=path, shapeType=letter)
        return shape



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
                            try:
                                data_stroke.normalize_wrt_max()
                            except ValueError:
                                print(name+" is empty")
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
            print("Loading ref data {}".format(name))
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
