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

new_demo = re.compile('demo:(?P<letter>.) "(?P<path>.*)')

##################### functions :

def load_learning_manager():

    fileName = inspect.getsourcefile(LearningManager)
    installDirectory = fileName.split('/lib')[0]
    robotDirectory = installDirectory + '/share/allograph/robot_tries/templates(non_cursive)'
    refDirectory = installDirectory + '/share/allograph/letter_model_dataset/alexis_set_for_children'

    return LearningManager(refDirectory,refDirectory,robotDirectory,refDirectory)

def load_log1(child):
    path =  '/home/alexis/Documents/cowriter_logs/EIntGen/visionLog_activity/'+child+'.csv'
    return path

def load_log2(child):
    path =  '/home/alexis/Documents/cowriter_logs/EIntGen/visionLog_activity/'+child+'_s2.csv'
    return path

def minScore(data, draw, name):
    try:
        score_min = stroke.cloud_dist(data[name][0],draw)
    except IndexError:
        draw.downsampleShape(70)
        score_min = stroke.cloud_dist(data[name][0],draw)
        #print data[name][0].get_len()
        #print draw.get_len()
    for ref in data[name]:
        try:
            score_new = stroke.cloud_dist(ref,draw)
        except IndexError:
            draw.downsampleShape(70)
            score_new = stroke.cloud_dist(ref,draw)
            #print ref.get_len()
            #print draw.get_len()
        if score_new<score_min:
            score_min = score_new
    return score_min

##################### MAIN SCRIPT :

if __name__ == "__main__":

    ############## logs analysis :

    learning_manager = load_learning_manager()

    letter_score = {}

    target = sys.argv[1]


    ############## letter_score for normalization

    for child in children_x2:
        log_path1 = load_log1(child)
        log_path2 = load_log2(child)

        i = 0
        with open(log_path1, 'r') as log:

            for line in log.readlines():

                found_demo = new_demo.search(line)

                if found_demo :#and opens[0]:

                    string = str(found_demo.group("path"))[:-1]
                    name = found_demo.group("letter")
                    try:
                        path = np.array(literal_eval(string.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        score = minScore(learning_manager.robot_data,graph,name)
                        if name in letter_score:
                            letter_score[name] += [score]
                        else:
                            letter_score[name] = [score]
                    except SyntaxError:
                        pass
        i = 0
        with open(log_path1, 'r') as log:
        
            for line in log.readlines():

                found_demo = new_demo.search(line)

                if found_demo :#and opens[0]:

                    string = str(found_demo.group("path"))[:-1]
                    name = found_demo.group("letter")
                    try:
                        path = np.array(literal_eval(string.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        score = minScore(learning_manager.robot_data,graph,name)
                        if name in letter_score:
                            letter_score[name] += [score]
                        else:
                            letter_score[name] = [score]
                    except SyntaxError:
                        pass


    for child in children_x1:
        log_path1 = load_log1(child)
        i = 0
        with open(log_path1, 'r') as log:
        
            for line in log.readlines():

                found_demo = new_demo.search(line)

                if found_demo :#and opens[0]:

                    string = str(found_demo.group("path"))[:-1]
                    name = found_demo.group("letter")
                    try:
                    
                        path = np.array(literal_eval(string.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        score = minScore(learning_manager.robot_data,graph,name)
                        if name in letter_score:
                            letter_score[name] += [score]
                        else:
                            letter_score[name] = [score]
                    except SyntaxError:
                        pass

    for name in letter_score:
        letter_score[name] = np.max(letter_score[name])

    print letter_score
        

    ############## child_score

    child_score = {}

    for child in children_x2:
        log_path1 = load_log1(child)
        log_path2 = load_log2(child)

        i = 0
        with open(log_path1, 'r') as log:

        
            for line in log.readlines():

                found_demo = new_demo.search(line)

                if found_demo :#and opens[0]:

                    string = str(found_demo.group("path"))[:-1]
                    name = found_demo.group("letter")
                    try:
                        
                        path = np.array(literal_eval(string.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        score = minScore(learning_manager.robot_data,graph,name)/letter_score[name]
                        if True:# name==target:
                            if child in child_score:
                                child_score[child] += [score]
                            else:
                                child_score[child] = [score]
                    except SyntaxError:
                        pass
        i = 0
        with open(log_path1, 'r') as log:
        
            for line in log.readlines():

                found_demo = new_demo.search(line)

                if found_demo :#and opens[0]:

                    string = str(found_demo.group("path"))[:-1]
                    name = found_demo.group("letter")
                    try:
                        path = np.array(literal_eval(string.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        score = minScore(learning_manager.robot_data,graph,name)/letter_score[name]
                        if True:# name==target:
                            if child in child_score:
                                child_score[child] += [score]
                            else:
                                child_score[child] = [score]
                    except SyntaxError:
                        pass


    for child in children_x1:
        log_path1 = load_log1(child)
        print child
        i = 0
        with open(log_path1, 'r') as log:
        
            for line in log.readlines():

                found_demo = new_demo.search(line)

                if found_demo :#and opens[0]:

                    string = str(found_demo.group("path"))[:-1]
                    name = found_demo.group("letter")
                    try:
                        path = np.array(literal_eval(string.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        score = minScore(learning_manager.robot_data,graph,name)/letter_score[name]
                        if True:# name==target:
                            if child in child_score:
                                child_score[child] += [score]
                            else:
                                child_score[child] = [score]
                        i+=1
                    except SyntaxError:
                        #print string
                        pass
        print i

    chouchou_score = []
    nonchouchou_score = []
    for child in chouchou:
        if child in child_score:
            chouchou_score.append(np.mean(child_score[child]))
    for child in nochouchou:
        if child in child_score:
            nonchouchou_score.append(np.mean(child_score[child]))

    print chouchou_score
    print nonchouchou_score

    print ""
    print ""
    print np.mean(chouchou_score)
    print np.var(chouchou_score)
    print len(chouchou_score)
    print ""
    print np.mean(nonchouchou_score)
    print np.var(nonchouchou_score)
    print len(nonchouchou_score)

    print np.mean(child_score['daniel'])
    print np.mean(child_score['dan'])
 

