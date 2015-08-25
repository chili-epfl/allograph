#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.cluster import MeanShift

import sys
import re
import inspect
import glob

from shape_learning.shape_modeler import ShapeModeler 
from allograph.learning_manager import LearningManager, stroke

import os.path

#################### FUNCTION FOR PROBA DENSITY :

def sigmo_powa(x, alpha, beta):
    return (1/(1+np.exp(-x+alpha)))**beta

#################### FUNCTION FOR EIGENSPACE :

NUM_PRINCIPLE_COMPONENTS = 10
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

#################### GET REF LETTERS :

fileName = inspect.getsourcefile(ShapeModeler)
installDirectory = fileName.split('/lib')[0]
refDirectory =  installDirectory + '/share/shape_learning/letter_model_datasets/bad_letters'

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


##################### SPARSER :


new_word = re.compile('2015-(?P<date>..-..) (?P<hour>..:..:..).*Received word: (?P<word>\w+)')
new_demo = re.compile('2015-(?P<date>..-..) (?P<hour>..:..:..).*Received template demonstration for letters')
new_demo_letter = re.compile('2015-(?P<date>..-..) (?P<hour>..:..:..).*Received demonstration for (?P<letter>\w)')
new_date = re.compile('2015-(?P<date>..-..) (?P<hour>..:..:..).*STATE: WAITING_FOR_FEEDBACK')
new_grade = re.compile('2015-(?P<date>..-..) (?P<hour>..:..:..).*User feedback (?P<grade>[+-])')

##################### Robot progress in a dict :

def load_learning_manager(child):
    fileName = inspect.getsourcefile(ShapeModeler)
    installDirectory = fileName.split('/lib')[0]
    robotDirectory = installDirectory + '/share/shape_learning/robot_tries/with_'+child

    return LearningManager(refDirectory,refDirectory,robotDirectory)


##################### SOME GLOBAL VALUES :

index={}
for letter in alphabet:
    if letter in "0123456789":
        index[letter]=0
    else:
        index[letter]=1

hour = ''
date = ''
cur_word = ''
cur_letter = ''
letter_score = -1
demo_done = False

mean_start = 0

progress = []
vals = []
max_progress = []
max_vals = []

scores = []
ch_scores = []

child_progress = []

grades = []

opens = np.zeros(4)

##################### MAIN SCRIPT :

if __name__ == "__main__":

    ############## logs analysis :

    ref_letters = read_ref_data(refDirectory, 6)
    child = sys.argv[2]
    learning_manager = load_learning_manager(child)


    i = 0
    with open(sys.argv[1], 'r') as log:
    
        for line in log.readlines():

            found_word = new_word.search(line)
            found_demo = new_demo.search(line)
            found_demo_letter = new_demo_letter.search(line)
            found_date = new_date.search(line)
            found_grade = new_grade.search(line)

            if found_word :#and opens[0]:

                correction=False
                if len(found_word.group('word'))==1:
                    if found_word.group('word') in cur_word:
                        correction=True
                ok = True
                for letter in set(cur_word):
                    if letter not in alphabet:
                        ok = False
                # look for single progression :
                if len(cur_word)>0 and ok:
                    new_scores = []
                    new_ch_wcores = []
                    for letter in set(cur_word):
                        #index[letter]+=1
                        i = index[letter]
                        try:
                            istroke = learning_manager.robot_data[letter][i]
                        except IndexError:
                            print "!"
                        last_stroke = learning_manager.robot_data[letter][i-1]
                        ch_stroke = last_stroke + (istroke-last_stroke)*2
                        _,dist = stroke.euclidian_distance(ch_stroke, ref_letters[letter])
                        _,ch_dist = stroke.euclidian_distance(ch_stroke, ref_letters[letter])
                        new_scores.append(ch_dist)
                        if letter==cur_letter:
                            letter_name = ch_dist

                    max_val = np.max(np.array(new_scores))
                    arg_max = np.argmax(np.array(new_scores))
                    old_val = scores[arg_max]
                    mean_val = np.mean(np.array(new_scores))

                    if opens[0]:
                        progress.append(start_mean - np.mean(np.array(scores)))

                    #print scores
                    #print new_scores
                    for i in range(len(new_scores)):
                        scores[i] = np.array(new_scores)[i]


                cur_word = found_word.group('word')
                scores = []#np.zeros(len(cur_word))
                ok = True
                for letter in set(cur_word):
                    if letter not in alphabet:
                        ok = False
                if len(cur_word)>0 and ok:
                    new_scores = []
                    for letter in cur_word:
                        try:
                            i = index[letter]
                        except KeyError:
                            print cur_word
                            print line
                            i = index[letter]
                        last_stroke = learning_manager.robot_data[letter][i-1]
                        istroke = learning_manager.robot_data[letter][i]
                        ch_stroke = last_stroke + (istroke-last_stroke)*2
                        _,dist = stroke.euclidian_distance(ch_stroke, ref_letters[letter])
                        scores.append(dist)
                    start_mean = np.mean(np.array(scores))
                    demo_done=False
                    opens[0]=0
                else:
                    start_mean = 0
                    opens = np.zeros(4)
                    opens[0]=1

            if found_demo and opens[1]:
                demo_done=True
                new_scores = []
                for letter in set(cur_word):
                    try:
                        index[letter]+=1
                    except KeyError:
                        print line
                        index[letter]+=1
                    i = index[letter]
                    try:
                        istroke = learning_manager.robot_data[letter][i]
                    except IndexError:
                        istroke = learning_manager.robot_data[letter][i-1]
                    last_stroke = learning_manager.robot_data[letter][i-1]
                    ch_stroke = last_stroke + (istroke-last_stroke)*2
                    _,dist = stroke.euclidian_distance(ch_stroke, ref_letters[letter])
                    new_scores.append(dist)
                    if letter==cur_letter:
                        letter_score = dist

                max_val = np.max(np.array(new_scores))
                arg_max = np.argmax(np.array(new_scores))
                old_val = scores[arg_max]
                mean_val = np.mean(np.array(new_scores))
                progress.append(start_mean-mean_val)

                for i in range(len(new_scores)):
                    scores[i] = np.array(new_scores)[i]
                opens[1]=0

            if found_demo_letter and opens[2]:
                demo_done=True

                letter = found_demo_letter.group('letter')
                if True:#letter==cur_letter:
                    if letter==cur_letter:
                        index[letter]+=1
                    else:
                        cur_letter=letter
                    i = index[letter]
                    try:
                        istroke = learning_manager.robot_data[letter][i]
                    except IndexError:
                        istroke = learning_manager.robot_data[letter][i-1]
                    last_stroke = learning_manager.robot_data[letter][i-1]
                    ch_stroke = last_stroke + (istroke-last_stroke)*2
                    _,dist = stroke.euclidian_distance(ch_stroke, ref_letters[letter])
                    for wletter,i in zip(cur_word,range(len(cur_word))):
                        if wletter==letter:
                            scores[i] = dist
                mean_val = np.mean(np.array(scores))
                # some scores are negatives
                progress.append(start_mean-mean_val)


                opens[2]=0

            if found_grade and opens[3]:
                if found_grade.group('grade')=='+':
                    #print found_grade.group('grade')
                    grades.append(1)
                else:
                    grades.append(-1)
                opens[3]=0


            if found_date:
                if opens[3]:
                    grades.append(0)

                date = found_date.group('date')
                hour = found_date.group('hour')
                nhour = ""
                if hour[6]=='0':
                    nhour+=hour[7]
                else:
                    nhour += hour[6:]
                if hour[3]=='0':
                    nhour = hour[4]+':'+nhour
                else:
                    nhour = hour[3]+hour[4]+':'+nhour
                if hour[0]=='0':
                    nhour = hour[1]+':'+nhour
                else:
                    nhour = hour[0]+hour[1]+':'+nhour

                hour = nhour.replace(':',',')
                hour = literal_eval('['+hour+']')
                opens = np.ones(4)

    if len(progress)>len(grades):
        progress=progress[:len(grades)]


    grades = np.array(grades)
    progress = np.array(progress)

    result = np.sum(grades*progress)

    # generating random vectors with same number of positive grades : 
    rand_results = []
    rand_grades = np.copy(grades)
    for i in range(100000):
        np.random.shuffle(rand_grades)
        rand_results.append(np.sum(rand_grades*progress))

    # supposing normal distribution :
    rand_results = np.array(rand_results)
    m = np.mean(rand_results)
    v = np.var(rand_results)
    
    z = (result-m)/np.sqrt(v)
    pvalue = 1 - 0.5*(1+erf(z/np.sqrt(2)))

    print pvalue
    print len(grades)
    print len(grades[grades>0])
    print len(grades[grades<0])

    plt.plot(progress,'r')
    plt.plot(grades,'b')
    plt.show()

