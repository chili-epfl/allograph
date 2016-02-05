#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.optimize import curve_fit
from scipy.special import erf
#from sklearn.cluster import MeanShift

import sys
import re
import inspect
import glob

from allograph.learning_manager import LearningManager
from allograph.stroke import Stroke
import allograph.stroke as stroke

import os.path

import csv
import copy
#################### globals :

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
children_x2 = {
        "alexandra","amelia","avery","daniel","enzo",
        "gaia","ines","jake","lamonie","markus",
        "osborne","oscar"
        }
children_x1 = {
        "dan","ege","jacqueline","kayra","laith",
        "lila","loulwa","ludovika","nadine","rin",
        "william","anais"
        }
chouchou = {
        "alexandra","avery","daniel",
        "gaia","ines","jake","lamonie","markus",
        "oscar"
        }
nochouchou = {
        "dan","ege","jacqueline","kayra","laith",
        "lila","loulwa","ludovika","nadine","rin",
        "william","anais","amelia","enzo","osborne"
        }


##################### sparser :

new_demo = re.compile('"(?P<time>..........................)" demo:(?P<letter>.) "(?P<path>.*)')
new_button = re.compile('user_feedback:(?P<button>.)')

##################### functions :

def load_log1(child):
    path =  '/home/alexis/Documents/cowriter_logs/EIntGen/visionLog_activity/'+child+'.csv'
    return path

def load_log2(child):
    path =  '/home/alexis/Documents/cowriter_logs/EIntGen/visionLog_activity/'+child+'_s2.csv'
    return path

##################### MAIN SCRIPT :

if __name__ == "__main__":


    ############## letter_score for normalization

    for child in children_x2:
        log_path1 = load_log1(child)
        log_path2 = load_log2(child)

        prev_strokes = {}

        i = 0
        feedback = 0
        with open(child+'.csv', 'wb') as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

            with open(log_path1, 'r') as log:

                for line in log.readlines():

                    found_demo = new_demo.search(line)
                    found_button = new_button.search(line)


                    new_feedback = 0
                    if found_button:
                        #print found_button.group("button")
                        #print child
                        if str(found_button.group("button"))=="+":
                            new_feedback = 1
                        else:
                            new_feedback = -1

                    if found_demo :
                        print feedback

                        string = str(found_demo.group("path"))[:-1]
                        name = found_demo.group("letter")
                        time = found_demo.group("time")
                        ischouchou = child in chouchou
                        try:
                            path = np.array(literal_eval(string.replace('"','')))
                            graph = Stroke()
                            graph.stroke_from_xxyy(path)
                            graph.downsampleShape(70)
                            
                            if name in prev_strokes:
                                _,score = stroke.euclidian_distance(prev_strokes[name], graph)
                                prev_strokes[name].reset()
                                prev_strokes[name].stroke_from_xxyy(path)
                                prev_strokes[name].downsampleShape(70)
                            else:
                                score = -1.
                                prev_strokes[name]=Stroke()
                                prev_strokes[name].stroke_from_xxyy(path)
                                prev_strokes[name].downsampleShape(70)

                            wr.writerow([time,child,name,score,ischouchou,feedback])
                        

                        except SyntaxError:
                            pass
                    if found_button or found_demo:
                        feedback = copy.copy(new_feedback)

