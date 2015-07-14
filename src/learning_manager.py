#!/usr/bin/env python

"""
learning allograph of letters from demonstration 
using different strategies and different metrics
"""

import logging; shapeLogger = logging.getLogger("shape_logger")
import os.path
from allograph import stroke

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

class LearningManager():

    def __init__(self, generate_path, demo_path, robot_path):
        self.generate_path = generate_path
        self.demo_path = demo_path
        self.robot_path = robot_path
        self.generate_letters('last_state')
        self.word_to_learn = ""

    def generate_word(self, word):
        generated_word = []
        for letter in word:
            generate_word.append(self.generated_letters[letter])
        return generated_word

    def generate_letters(self, mode='last_state'):
        if mode == 'last_state':
            for letter in alphabet:
                stroke = self.read_last_line(letter)
                self.generated_letters[letter] = stroke
        #if mode = 'PCA' 
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)

    def respond_to_demonstration(self, demonstration, mode='midway'): #mutual_modeling will act here
        if mode == 'midway':
            for letter,stroke in demonstration:
                learned_stroke = stroke.midway(stroke, self.generated_letters[letter])
                self.generated_letters[letter] = learned_stroke
        #if mode = 'PCA' 
        #if mode = 'sigNorm' (mixture of sigma-log-normal)
        #if mode = 'CNN' (1-D convolutionnal neural networks)

    def read_last_line(self, letter):





