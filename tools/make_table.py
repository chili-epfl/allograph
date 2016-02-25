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
import math

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

start_f2f = {
        "alexandra","larmonie","ines","markus",
        "oscar","ege","daniel","laith","kayra",
        "wiliam","rin","ludovica","loulwa",
        "anais","jacqueline","nadine","dan"
        }

##################### sparser :

new_demo = re.compile('"(?P<time>..........................)" demo:(?P<letter>.) "(?P<path>\(.*\))')
new_demo_bug = re.compile('"(?P<time>..........................)" demo:(?P<letter>.) "(?P<path1>\(.*)"2016-')
new_demo_bug2 = re.compile('(?P<path2>.*\))')

new_learn = re.compile('"(?P<time>..........................)" learn:(?P<letter>.) "(?P<path>\(.*\))')

new_button = re.compile('user_feedback:(?P<button>.)')
new_word = re.compile('word:(?P<word>.*)')
new_repetition = re.compile('repetition:(?P<number>.)')

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
    with open('learns.csv', 'wb') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)


        for child in children_x2:
            log_path1 = load_log1(child)
            log_path2 = load_log2(child)

            prev_strokes = {}

            i = 0
            feedback = float('nan')
            new_time = ""
            new_child = ""
            new_name = ""
            new_score = 0
            new_ischouchou = False
            
            word_done = False
            bug_on_last_line = False
            string_bug = ""
            bug_variables = []

            lines = []
            f2f = False

            current_learn = {}
            current_demo = {}

            nb_fb=1.


            with open(log_path1, 'r') as log:

                if child in start_f2f:
                    f2f = True

                for line in log.readlines():

                    found_demo = new_demo.search(line)
                    found_button = new_button.search(line)
                    found_word = new_word.search(line)
                    found_repetition = new_repetition.search(line)
                    found_demo_bug = new_demo_bug.search(line)
                    found_demo_bug2 = new_demo_bug2.search(line)
                    found_learn = new_learn.search(line)

                    if found_repetition:
                        word_done = True
                        current_letters = {}

                    if found_learn:
                        string = str(found_learn.group("path"))#[:-1]
                        name = found_learn.group("letter")
                        
                        path = np.array(literal_eval(string))#.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        graph.downsampleShape(70)
                        current_learn[name]=graph



                    if found_word:
                        word = str(found_word.group("word"))[:-1]
                        if word_done:
                            for liste in lines:
                                letter = liste[2]
                                if letter in current_demo and letter in current_learn:
                                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,1,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
                            lines = []
                            current_demo = {}
                            current_learn = {}
                            feedback = float('nan')
                            word_done = False
                            nb_fb=1.

                    if found_button:
                        
                        
                        if str(found_button.group("button"))=="+":
                            if math.isnan(feedback):
                                feedback = 1
                            else:
                                feedback += 1
                                nb_fb+=1
                        else:
                            if math.isnan(feedback):
                                feedback = -1
                            else:
                                feedback -= 1
                                nb_fb+=1

                    if bug_on_last_line:
                        time = bug_variables[0]
                        name = bug_variables[1]
                        ischouchou = bug_variables[2]

                        if found_demo_bug2:
                            string2 = str(found_demo_bug2.group("path2"))
                            string = string_bug+string2
                            try:
                                path = np.array(literal_eval(string))#.replace('"','')))
                                graph = Stroke()
                                graph.stroke_from_xxyy(path)
                                graph.downsampleShape(70)
                                current_demo[name] = graph

                                liste = []
                                liste.append(time)
                                liste.append(child)
                                liste.append(name)
                                liste.append(ischouchou)
                                lines.append(liste)
                            except SyntaxError:
                                try:
                                    path = np.array(literal_eval(string_bug+')'))#.replace('"','')))
                                    graph = Stroke()
                                    graph.stroke_from_xxyy(path)
                                    graph.downsampleShape(70)
                                    current_demo[name] = graph
                                    liste = []
                                    liste.append(time)
                                    liste.append(child)
                                    liste.append(name)
                                    liste.append(ischouchou)
                                    lines.append(liste)
                                except SyntaxError:
                                    if child=='oscar':
                                        print 'second_bug_not_translated'
                                        print string_bug
                                        print string2
                        else:
                            try:
                                path = np.array(literal_eval(string_bug+')'))#.replace('"','')))
                                graph = Stroke()
                                graph.stroke_from_xxyy(path)
                                graph.downsampleShape(70)
                                current_demo[name] = graph
                                liste = []
                                liste.append(time)
                                liste.append(child)
                                liste.append(name)
                                liste.append(ischouchou)
                                lines.append(liste)
                            except SyntaxError:
                                if child=='oscar':
                                    print 'last_bug_not_translated'
                                    print line

                        bug_on_last_line = False
                        bug_variables = []
    


                    if found_demo :

                        if word_done:
                            for liste in lines:
                                letter = liste[2]
                                if letter in current_demo and letter in current_learn:
                                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,1,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
                            lines = []
                            current_demo = {}
                            current_learn = {}
                            feedback = float('nan')
                            word_done = False
                            nb_fb=1.
                        
                        #wr.writerow([new_time,new_child,new_name,new_score,new_ischouchou,feedback])

                        string = str(found_demo.group("path"))#[:-1]
                        name = found_demo.group("letter")
                        time = found_demo.group("time")
                        ischouchou = child in chouchou
                        try:
                            path = np.array(literal_eval(string))#.replace('"','')))
                            graph = Stroke()
                            graph.stroke_from_xxyy(path)
                            graph.downsampleShape(70)
                            current_demo[name] = graph
                            liste = []
                            liste.append(time)
                            liste.append(child)
                            liste.append(name)
                            liste.append(ischouchou)
                            lines.append(liste)
                            #wr.writerow([time,child,name,score,ischouchou,feedback])
                        

                        except SyntaxError:
                            if found_demo_bug:
                                string_bug = str(found_demo_bug.group("path1"))
                                bug_on_last_line = True
                                bug_variables.append(time)
                                bug_variables.append(name)
                                bug_variables.append(ischouchou)

                            else:
                                if child=='oscar':
                                    print "bug not found"
                                    print string





                    #if found_button or found_word or found_repetition:
                    #    feedback = copy.copy(new_feedback)

            #wr.writerow([new_time,new_child,new_name,new_score,new_ischouchou,feedback])
            for liste in lines:
                letter = liste[2]
                if letter in current_demo and letter in current_learn:
                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,1,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
            lines = []
            current_demo = {}
            current_learn = {}

            prev_strokes = {}

            i = 0
            feedback = float('nan')
            new_time = ""
            new_child = ""
            new_name = ""
            new_score = 0
            new_ischouchou = False
            
            word_done = False
            bug_on_last_line = False
            string_bug = ""
            bug_variables = []

            lines = []
            f2f = True

            current_learn = {}
            current_demo = {}

            nb_fb=1.


            with open(log_path2, 'r') as log:

                if child in start_f2f:
                    f2f = False

                for line in log.readlines():

                    found_demo = new_demo.search(line)
                    found_button = new_button.search(line)
                    found_word = new_word.search(line)
                    found_repetition = new_repetition.search(line)
                    found_demo_bug = new_demo_bug.search(line)
                    found_demo_bug2 = new_demo_bug2.search(line)
                    found_learn = new_learn.search(line)

                    if found_repetition:
                        word_done = True
                        current_letters = {}

                    if found_learn:
                        string = str(found_learn.group("path"))#[:-1]
                        name = found_learn.group("letter")
                        
                        path = np.array(literal_eval(string))#.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        graph.downsampleShape(70)
                        current_learn[name]=graph



                    if found_word:
                        word = str(found_word.group("word"))[:-1]
                        if word_done:
                            for liste in lines:
                                letter = liste[2]
                                if letter in current_demo and letter in current_learn:
                                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,2,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
                            lines = []
                            current_demo = {}
                            current_learn = {}
                            feedback = float('nan')
                            word_done = False
                            nb_fb=1.

                    if found_button:
                        
                        
                        if str(found_button.group("button"))=="+":
                            if math.isnan(feedback):
                                feedback = 1
                            else:
                                feedback += 1
                                nb_fb+=1
                        else:
                            if math.isnan(feedback):
                                feedback = -1
                            else:
                                feedback -= 1
                                nb_fb+=1

                    if bug_on_last_line:
                        time = bug_variables[0]
                        name = bug_variables[1]
                        ischouchou = bug_variables[2]

                        if found_demo_bug2:
                            string2 = str(found_demo_bug2.group("path2"))
                            string = string_bug+string2
                            try:
                                path = np.array(literal_eval(string))#.replace('"','')))
                                graph = Stroke()
                                graph.stroke_from_xxyy(path)
                                graph.downsampleShape(70)
                                current_demo[name] = graph

                                liste = []
                                liste.append(time)
                                liste.append(child)
                                liste.append(name)
                                liste.append(ischouchou)
                                lines.append(liste)
                            except SyntaxError:
                                try:
                                    path = np.array(literal_eval(string_bug+')'))#.replace('"','')))
                                    graph = Stroke()
                                    graph.stroke_from_xxyy(path)
                                    graph.downsampleShape(70)
                                    current_demo[name] = graph
                                    liste = []
                                    liste.append(time)
                                    liste.append(child)
                                    liste.append(name)
                                    liste.append(ischouchou)
                                    lines.append(liste)
                                except SyntaxError:
                                    if child=='oscar':
                                        print 'second_bug_not_translated'
                                        print string_bug
                                        print string2
                        else:
                            try:
                                path = np.array(literal_eval(string_bug+')'))#.replace('"','')))
                                graph = Stroke()
                                graph.stroke_from_xxyy(path)
                                graph.downsampleShape(70)
                                current_demo[name] = graph
                                liste = []
                                liste.append(time)
                                liste.append(child)
                                liste.append(name)
                                liste.append(ischouchou)
                                lines.append(liste)
                            except SyntaxError:
                                if child=='oscar':
                                    print 'last_bug_not_translated'
                                    print line

                        bug_on_last_line = False
                        bug_variables = []
    


                    if found_demo :

                        if word_done:
                            for liste in lines:
                                letter = liste[2]
                                if letter in current_demo and letter in current_learn:
                                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,2,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
                            lines = []
                            current_demo = {}
                            current_learn = {}
                            feedback = float('nan')
                            word_done = False
                            nb_fb=1.
                        
                        #wr.writerow([new_time,new_child,new_name,new_score,new_ischouchou,feedback])

                        string = str(found_demo.group("path"))#[:-1]
                        name = found_demo.group("letter")
                        time = found_demo.group("time")
                        ischouchou = child in chouchou
                        try:
                            path = np.array(literal_eval(string))#.replace('"','')))
                            graph = Stroke()
                            graph.stroke_from_xxyy(path)
                            graph.downsampleShape(70)
                            current_demo[name] = graph
                            liste = []
                            liste.append(time)
                            liste.append(child)
                            liste.append(name)
                            liste.append(ischouchou)
                            lines.append(liste)
                            #wr.writerow([time,child,name,score,ischouchou,feedback])
                        

                        except SyntaxError:
                            if found_demo_bug:
                                string_bug = str(found_demo_bug.group("path1"))
                                bug_on_last_line = True
                                bug_variables.append(time)
                                bug_variables.append(name)
                                bug_variables.append(ischouchou)

                            else:
                                if child=='oscar':
                                    print "bug not found"
                                    print string





                    #if found_button or found_word or found_repetition:
                    #    feedback = copy.copy(new_feedback)

            #wr.writerow([new_time,new_child,new_name,new_score,new_ischouchou,feedback])
            for liste in lines:
                letter = liste[2]
                if letter in current_demo and letter in current_learn:
                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,2,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
            lines = []
            current_demo = {}
            current_learn = {}

    #with open('1sess.csv', 'wb') as csvfile:
        #wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        for child in children_x1:
            log_path1 = load_log1(child)

            prev_strokes = {}

            i = 0
            feedback = float('nan')
            new_time = ""
            new_child = ""
            new_name = ""
            new_score = 0
            new_ischouchou = False
            
            word_done = False
            bug_on_last_line = False
            string_bug = ""
            bug_variables = []

            lines = []
            f2f = False

            current_learn = {}
            current_demo = {}

            nb_fb=1.


            with open(log_path1, 'r') as log:

                if child in start_f2f:
                    f2f = True

                for line in log.readlines():

                    found_demo = new_demo.search(line)
                    found_button = new_button.search(line)
                    found_word = new_word.search(line)
                    found_repetition = new_repetition.search(line)
                    found_demo_bug = new_demo_bug.search(line)
                    found_demo_bug2 = new_demo_bug2.search(line)
                    found_learn = new_learn.search(line)

                    if found_repetition:
                        word_done = True
                        current_letters = {}

                    if found_learn:
                        string = str(found_learn.group("path"))#[:-1]
                        name = found_learn.group("letter")
                        
                        path = np.array(literal_eval(string))#.replace('"','')))
                        graph = Stroke()
                        graph.stroke_from_xxyy(path)
                        graph.downsampleShape(70)
                        current_learn[name]=graph



                    if found_word:
                        word = str(found_word.group("word"))[:-1]
                        if word_done:
                            for liste in lines:
                                letter = liste[2]
                                if letter in current_demo and letter in current_learn:
                                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,1,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
                            lines = []
                            current_demo = {}
                            current_learn = {}
                            feedback = float('nan')
                            word_done = False
                            nb_fb=1.

                    if found_button:
                        
                        
                        if str(found_button.group("button"))=="+":
                            if math.isnan(feedback):
                                feedback = 1
                            else:
                                feedback += 1
                                nb_fb+=1
                        else:
                            if math.isnan(feedback):
                                feedback = -1
                            else:
                                feedback -= 1
                                nb_fb+=1

                    if bug_on_last_line:
                        time = bug_variables[0]
                        name = bug_variables[1]
                        ischouchou = bug_variables[2]

                        if found_demo_bug2:
                            string2 = str(found_demo_bug2.group("path2"))
                            string = string_bug+string2
                            try:
                                path = np.array(literal_eval(string))#.replace('"','')))
                                graph = Stroke()
                                graph.stroke_from_xxyy(path)
                                graph.downsampleShape(70)
                                current_demo[name] = graph

                                liste = []
                                liste.append(time)
                                liste.append(child)
                                liste.append(name)
                                liste.append(ischouchou)
                                lines.append(liste)
                            except SyntaxError:
                                try:
                                    path = np.array(literal_eval(string_bug+')'))#.replace('"','')))
                                    graph = Stroke()
                                    graph.stroke_from_xxyy(path)
                                    graph.downsampleShape(70)
                                    current_demo[name] = graph
                                    liste = []
                                    liste.append(time)
                                    liste.append(child)
                                    liste.append(name)
                                    liste.append(ischouchou)
                                    lines.append(liste)
                                except SyntaxError:
                                    if child=='oscar':
                                        print 'second_bug_not_translated'
                                        print string_bug
                                        print string2
                        else:
                            try:
                                path = np.array(literal_eval(string_bug+')'))#.replace('"','')))
                                graph = Stroke()
                                graph.stroke_from_xxyy(path)
                                graph.downsampleShape(70)
                                current_demo[name] = graph
                                liste = []
                                liste.append(time)
                                liste.append(child)
                                liste.append(name)
                                liste.append(ischouchou)
                                lines.append(liste)
                            except SyntaxError:
                                if child=='oscar':
                                    print 'last_bug_not_translated'
                                    print line

                        bug_on_last_line = False
                        bug_variables = []
    


                    if found_demo :

                        if word_done:
                            for liste in lines:
                                letter = liste[2]
                                if letter in current_demo and letter in current_learn:
                                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,1,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
                            lines = []
                            current_demo = {}
                            current_learn = {}
                            feedback = float('nan')
                            word_done = False
                            nb_fb=1.
                        
                        #wr.writerow([new_time,new_child,new_name,new_score,new_ischouchou,feedback])

                        string = str(found_demo.group("path"))#[:-1]
                        name = found_demo.group("letter")
                        time = found_demo.group("time")
                        ischouchou = child in chouchou
                        try:
                            path = np.array(literal_eval(string))#.replace('"','')))
                            graph = Stroke()
                            graph.stroke_from_xxyy(path)
                            graph.downsampleShape(70)
                            current_demo[name] = graph
                            liste = []
                            liste.append(time)
                            liste.append(child)
                            liste.append(name)
                            liste.append(ischouchou)
                            lines.append(liste)
                            #wr.writerow([time,child,name,score,ischouchou,feedback])
                        

                        except SyntaxError:
                            if found_demo_bug:
                                string_bug = str(found_demo_bug.group("path1"))
                                bug_on_last_line = True
                                bug_variables.append(time)
                                bug_variables.append(name)
                                bug_variables.append(ischouchou)

                            else:
                                if child=='oscar':
                                    print "bug not found"
                                    print string





                    #if found_button or found_word or found_repetition:
                    #    feedback = copy.copy(new_feedback)

            #wr.writerow([new_time,new_child,new_name,new_score,new_ischouchou,feedback])
            for liste in lines:
                letter = liste[2]
                if letter in current_demo and letter in current_learn:
                    _,score = stroke.euclidian_distance(current_demo[letter], current_learn[letter])
                    wr.writerow([liste[0],liste[1],liste[2],liste[3],feedback/nb_fb,1,f2f,str(current_learn[letter].x),str(current_learn[letter].y)])
            lines = []
            current_demo = {}
            current_learn = {}


