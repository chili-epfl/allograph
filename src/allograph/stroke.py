#!/usr/bin/env python
# coding: utf-8

"""
library of algorithms to compare hand-written strokes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class Stroke:
    """ a stroke object is a collection of x coordinates and y coordinates
     that describes the points inside a 1-stroke shape """

    def __init__(self, x=[], y=[]):
        self.x = x
        self.y = y

        # self.len is the number of points in the stroke
        self.len = min(len(self.x), len(self.y))

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_len(self):
        return self.len

    def append(self, x, y):
        """ add the point (x,y) to the stroke """

        self.x.append(x)
        self.y.append(y)
        self.len += 1

    def __add__(self, stroke2):
        x1 = np.copy(np.array(self.x))
        x2 = np.array(stroke2.x)
        y1 = np.copy(np.array(self.y))
        y2 = np.array(stroke2.y)
        return Stroke(x1 + x2, y1 + y2)

    def __sub__(self, stroke2):
        x1 = np.copy(np.array(self.x))
        x2 = np.array(stroke2.x)
        y1 = np.copy(np.array(self.y))
        y2 = np.array(stroke2.y)
        return Stroke(x1 - x2, y1 - y2)

    def __mul__(self, num):
        x1 = np.copy(np.array(self.x))
        y1 = np.copy(np.array(self.y))
        return Stroke(x1 * num, y1 * num)
        
    def strokeToImage(self, dimension):
		
		if ( dimension <= 0 ):
			raise ValueError('dimension should be strictly positive')
			
		image = np.zeros(shape=(dimension,dimension))
	#~ scale = max((max(self.get_x())-min(self.get_x())),(max(self.get_y())-min(self.get_y())))
	#~ print scale
	#~ scaleFactor = dimension / scale
	#~ print scaleFactor
		"""Normalize according to the greatest dimension"""
		self.normalize_wrt_max()
	
		"""Initialization of the variables to save the last values"""
		prev_i = 0.0
		prev_j = 0.0
		"""Used for the case where prev does not exist"""
		first = True
		
		for i,j in zip(self.get_x(),self.get_y()):
			#~ i = int(i*scaleFactor)
			#~ j = int(i*scaleFactor)
			"""adjust i and j according to the dimension"""
			i = int(i*(dimension-1))
			j = int(j*(dimension-1))
			#~ print i 
			#~ print j
			"""fills the pixel"""
			image[j,i] = 1
	    
			"""if first do not consider last value"""
			if (first):
				first = False
				prev_i = i
				prev_j = j
				continue
				
			"""compute the difference between current and last value"""
			diff_i = i - prev_i
			diff_j = j - prev_j
	    
			if (abs(diff_j) >= abs(diff_i)):
				current_j = prev_j
				if (diff_i != 0):
					ratio = int(diff_j/abs(diff_i))
					if (ratio == 0):
						continue
					sign_i = diff_i/abs(diff_i)
					for col in range(prev_i, i, sign_i):
						for row in range(current_j,current_j+ratio, ratio/abs(ratio)):
							image[row,col] = 1
						current_j += ratio;
				else:
					if (diff_j == 0):
						continue
					ratio = diff_j
					for row in range(current_j,current_j + ratio, ratio/abs(ratio)):
						image[row,i] = 1
		
			
			else:
				current_i = prev_i
				if(diff_j != 0):
					ratio = int(diff_i/abs(diff_j))
					if (ratio == 0):
						continue
					sign_j = diff_j/abs(diff_j)
					for row in range(prev_j, j, sign_j):
						for col in range(current_i, current_i+ratio, ratio/abs(ratio)):
							image[row,col] = 1
						current_i += ratio
				else:
					if (diff_i == 0):
						continue
					ratio = diff_i
					for col in range(current_i, current_i+ratio, ratio/abs(ratio)):
						image[j,col] = 1
		
		    
			prev_i = i
			prev_j = j
		return image
		    
    def reset(self):
        self.x = []
        self.y = []
        self.len = 0

    def copy(self):
        x = np.copy(np.array(self.x))
        y = np.copy(np.array(self.y))
        return Stroke(x, y)

    def plot(self):
        plt.plot(self.x, -np.array(self.y), 'b')
        plt.show()
        # plt.plot(self.x,self.y,'r.')

    def multi_strokes_plot(self):
        # self.downsampleShape(70)
        strokes = []

        dists = []
        path = zip(self.x, self.y)
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            dists.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        mean_dist = np.mean(dists)
        density = np.array(dists) / mean_dist

        current_stroke = Stroke().copy()
        current_stroke.x = []
        current_stroke.y = []
        for i in range(len(self.x) - 1):
            current_stroke.append(self.x[i], self.y[i])
            if density[i] > 3.5:
                strokes.append(current_stroke.copy())
                current_stroke.reset()
        current_stroke.append(self.x[-1], self.y[-1])
        strokes.append(current_stroke.copy())

        for stroke in strokes:
            stroke.plot()

    def stroke_from_xxyy(self, shape):
        self.x = shape[0:len(shape) / 2]
        self.y = shape[len(shape) / 2:]
        self.len = len(self.x)

    def downsampleShape(self, numDesiredPoints):
        """ change the length of a stroke with interpolation"""

        if len(self.x) > 2:
            t_current_x = np.linspace(0, 1, len(self.x))
            t_current_y = np.linspace(0, 1, len(self.y))
            t_desired_x = np.linspace(0, 1, numDesiredPoints)
            t_desired_y = np.linspace(0, 1, numDesiredPoints)
            f = interpolate.interp1d(t_current_x, self.x, kind='linear')
            self.x = f(t_desired_x).tolist()
            f = interpolate.interp1d(t_current_y, self.y, kind='linear')
            self.y = f(t_desired_y).tolist()

            self.len = numDesiredPoints

    def euclidian_length(self):
        """ comput length of the shape """

        if self.get_len() > 1:
            shape_length = 0
            last_x = self.x
            last_y = self.y
            scale = [0]
            for i in range(self.len - 2):
                x = np.array(self.x[i + 1])
                y = np.array(self.y[i + 1])
                last_x = np.array(self.x[i])
                last_y = np.array(self.y[i])
                shape_length += np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                scale.append(shape_length)
            return shape_length, scale

        else:
            return 0, [0]

    def uniformize(self):
        """make the distribution of points in the stroke equidistants """

        self.len = len(self.x)

        if self.len > 1:
            # comput length of the shape:
            shape_length, scale = self.euclidian_length()

            # find new points:
            new_shape = Stroke()
            new_shape.x = []
            new_shape.y = []
            step = shape_length / float(self.len)
            biggest_smoller_point = 0
            new_shape.append(self.x[0], self.y[0])
            for i in 1 + np.array(range(len(self.x) - 1)):
                try:
                    while i * step > scale[biggest_smoller_point]:
                        biggest_smoller_point += 1

                    biggest_smoller_point -= 1
                    x0 = self.x[biggest_smoller_point]
                    y0 = self.y[biggest_smoller_point]
                    x1 = self.x[biggest_smoller_point + 1]
                    y1 = self.y[biggest_smoller_point + 1]
                    diff = float(i * step - scale[biggest_smoller_point])
                    dist = float(scale[biggest_smoller_point + 1] - scale[biggest_smoller_point])
                    new_x = x0 + diff * (x1 - x0) / dist
                    new_y = y0 + diff * (y1 - y0) / dist
                    new_shape.append(new_x, new_y)

                except IndexError:
                    print i * step
                    print biggest_smoller_point
                    print scale
            # new_shape.append(self.x[-1], self.y[-1])


            self.x = new_shape.x
            self.y = new_shape.y
            self.len = new_shape.len

    def revert(self):
        """ revert a stroke : [x1,x2,x3] --> [x3,x2,x1] """

        x = self.x[::-1]
        y = self.y[::-1]
        return Stroke(x, y)

    def get_center(self):
        """ compute the gravity center of a stroke """

        x = np.array(self.x)
        y = np.array(self.y)
        return np.mean(x), np.mean(y)

    def normalize(self):
        """ normalize the stroke """

        x_min = min(self.x)
        x_max = max(self.x)
        y_min = min(self.y)
        y_max = max(self.y)

        x_range = x_max - x_min
        y_range = y_max - y_min

        x = np.array(self.x)
        y = np.array(self.y)
        x -= x_min
        y -= y_min
        x = x / float(x_range)
        y = y / float(y_range)

        self.x = x.tolist()
        self.y = y.tolist()

    def normalize_wrt_x(self):
        """ normalize the stroke with respect to the x axis """

        x_min = min(self.x)
        x_max = max(self.x)
        y_min = min(self.y)

        x_range = x_max - x_min

        x = np.array(self.x)
        y = np.array(self.y)
        x -= x_min
        y -= y_min
        x = x / float(x_range)
        y = y / float(x_range)

        self.x = x.tolist()
        self.y = y.tolist()

    def normalize_wrt_y(self):
        """ normalize the stroke with respect to the x axis """

        x_min = min(self.x)
        y_min = min(self.y)
        y_max = max(self.y)

        y_range = y_max - y_min

        x = np.array(self.x)
        y = np.array(self.y)
        x -= x_min
        y -= y_min
        x = x / float(y_range)
        y = y / float(y_range)

        self.x = x.tolist()
        self.y = y.tolist()

    def normalize_wrt_max(self):
        """ normalize the stroke with respect to the x axis """

        x_min = min(self.x)
        x_max = max(self.x)
        y_min = min(self.y)
        y_max = max(self.y)

        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)

        x = np.array(self.x)
        y = np.array(self.y)
        x -= x_min
        y -= y_min
        x = x / float(max_range)
        y = y / float(max_range)

        self.x = x.tolist()
        self.y = y.tolist()
        
    def split_non_differentiable_points(self, treshold=1.5):
        """ V --> \+/ """

        # split_points = Stroke()
        splited_strokes = []
        current_stroke = Stroke()
        if len(self.x) > 3:
            current_stroke.append(self.x[0], self.y[0])
            for i in range(len(self.x) - 3):
                x1 = float(self.x[i])
                x2 = float(self.x[i + 1])
                x3 = float(self.x[i + 2])
                y1 = float(self.y[i])
                y2 = float(self.y[i + 1])
                y3 = float(self.y[i + 2])
                triangle_ratio = (np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + np.sqrt(
                    (x3 - x2) ** 2 + (y3 - y2) ** 2)) / (np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) + 0.0001)
                current_stroke.append(x2, y2)
                if triangle_ratio > treshold:
                    splited_strokes.append(current_stroke)
                    current_stroke.reset()
            if current_stroke.get_x():
                splited_strokes.append(current_stroke)

            return splited_strokes
        else:
            return [self]

        def split_by_density(self, treshold=3):
            """H -> |-| """

        splited_strokes = []
        current_stroke = Stroke()

        stroke_length, _ = self.euclidian_length()

        mean_dist = stroke_length / (self.get_len() + 0.000001)

        if len(self.x) > 3:
            current_stroke.append(self.x[0], self.y[0])
            for i in range(len(self.x) - 2):
                x1 = float(self.x[i])
                x2 = float(self.x[i + 1])
                y1 = float(self.y[i])
                y2 = float(self.y[i + 1])
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                density = dist / mean_dist
                current_stroke.append(x2, y2)
                if density > 3:
                    splited_strokes.append(current_stroke)
                    current_stroke.reset()
            if current_stroke.get_x():
                splited_strokes.append(current_stroke)

            return splited_strokes
        else:
            return [self]
   


# static functions:
# ------------------
def childFromRobot(RobotStrokes):
    childStrokes = []
    currentStroke = Stroke()
    for i in range(len(RobotStrokes) - 1):
        currentStroke = RobotStrokes[i + 1].__sub__(RobotStrokes[i])
        currentStroke = currentStroke.__mul__(2)
        currentStroke = currentStroke.__add__(RobotStrokes[i])
        childStrokes.append(currentStroke)
    return childStrokes


def strokeToArray(aStroke):
    return (np.append(np.array(aStroke.x), np.array(aStroke.y)))
	

def plot_list(strokes):
    length = len(strokes)
    for i in range(length):
        plt.subplot(1, length, i)
        strokes[i].plot()
        # plt.show()


def save_plot_list(strokes, name):
    length = len(strokes)
    for i in range(length):
        strokes[i].plot()
        indexed_name = '%s_%i.svg' % (name, i)
        plt.savefig(indexed_name)
        plt.clf()


def midway(stroke1, stroke2, coef=0):
    x1 = np.array(stroke1.x)
    x2 = np.array(stroke2.x)
    y1 = np.array(stroke1.y)
    y2 = np.array(stroke2.y)

    mu = 1. / (1 + np.exp(coef))
    nu = 1 - mu

    x = (mu * x1 + nu * x2)
    y = (mu * y1 + nu * y2)
    return Stroke(x, y)


def smart_split(strokes):
    """ split at non-differentiable points of each stroke of 'strokes' """

    splited = []
    for stroke in strokes:
        splited += stroke.split_non_differentiable_points()
    return splited


def concat(strokes):
    """ concatenate all the strokes of a multistroke drawing """

    long_stroke = Stroke()
    for stroke in strokes:
        long_stroke.x += stroke.x
        long_stroke.y += stroke.y
        long_stroke.len += stroke.len
    return long_stroke


def smart_concat(strokes):
    """ if a stroke starts where another ends --> concat """

    # compute the average of spaces between connected points
    total_length = 0
    for stroke in strokes:
        length, _ = stroke.euclidian_length()
        total_length += length
    space = total_length / float(len(strokes) * 7 + 0.00001)

    i = 0
    while len(strokes[i + 1:]) > 0:
        test = False
        stroke = strokes[i]
        indice = i
        for stroke_i in strokes[i + 1:]:

            indice += 1
            x11 = stroke.get_x()[-1]
            x21 = stroke_i.get_x()[0]
            y11 = stroke.get_y()[-1]
            y21 = stroke_i.get_y()[0]

            x12 = stroke.get_x()[-2]
            x22 = stroke_i.get_x()[1]
            y12 = stroke.get_y()[-2]
            y22 = stroke_i.get_y()[1]
            dist = np.sqrt((x11 - x21) ** 2 + (y11 - y21) ** 2)

            dx1 = x11 - x12
            dx2 = x21 - x11
            dx3 = x22 - x21
            dx = [dx1, dx2, dx3]
            # print dx

            dy1 = y11 - y12
            dy2 = y21 - y11
            dy3 = y22 - y21
            dy = [dy1, dy2, dy3]
            # print dy

            good_angle = False
            """if dx1*dx3<0 and dy1*dy3<0:
                good_angle = True"""

            if max(dx) * min(dx) > -10 and max(dy) * min(dy) > -10:
                good_angle = True

            if dist < space and good_angle:
                test = True
                strokes[i] = concat([stroke, stroke_i])
                strokes[indice:-1] = strokes[indice + 1:]
                strokes = strokes[:-1]
                break
        if not test:
            i += 1

    return strokes


def smart_merging(strokes, threshold=0.05):
    """ if we draw two times the same shape at the same place, forget the smaller one """

    new_strokes = []
    while len(strokes) > 1:
        test = True
        indice = 0
        for stroke in strokes[1:]:
            indice += 1
            score1 = identify([stroke], strokes[0])
            score2 = identify([strokes[0]], stroke)
            score = min(score1, score2)
            length1, _ = stroke.euclidian_length()
            length2, _ = strokes[0].euclidian_length()
            score = score / (min(length1, length2) + 0.0001)
            if score < threshold:
                test = False
                break
        if test:  # that means the stroke has'nt any twin
            copy = Stroke(strokes[0].get_x(), strokes[0].get_y())
            new_strokes.append(copy)
        else:
            # we want to keep the bigger:
            size1 = strokes[indice].euclidian_length()
            size2 = strokes[0].euclidian_length()
            if size2 > size1:
                strokes[indice] = strokes[0]

        strokes = strokes[1:]

    # add the last one (we deleted all his possible twins)
    new_strokes.append(strokes[0])
    return new_strokes


def group_normalize(strokes):
    """ normilize a multistroke drawing """

    long_stroke = concat(strokes)
    x_min = min(long_stroke.x)
    x_max = max(long_stroke.x)
    y_min = min(long_stroke.y)
    y_max = max(long_stroke.y)
    x_range = float(x_max - x_min)
    y_range = float(y_max - y_min)
    normalized_strokes = []
    for stroke in strokes:
        x = ((np.array(stroke.x) - x_min) / x_range).tolist()
        y = ((np.array(stroke.y) - y_min) / y_range).tolist()
        normalized_strokes.append(Stroke(x, y))
    return normalized_strokes


def group_normalize_wrt_x(strokes):
    """ normailize a multistroke drawing with respect to the x axis """

    long_stroke = concat(strokes)
    x_min = min(long_stroke.x)
    x_max = max(long_stroke.x)
    y_min = min(long_stroke.y)
    x_range = float(x_max - x_min)
    normalized_strokes = []
    for stroke in strokes:
        x = ((np.array(stroke.x) - x_min) / x_range).tolist()
        y = ((np.array(stroke.y) - y_min) / x_range).tolist()
        normalized_strokes.append(Stroke(x, y))
    return normalized_strokes


def group_normalize_wrt_y(strokes):
    """ normalize a multistroke drawing with respect to the x axis """

    long_stroke = concat(strokes)
    x_min = min(long_stroke.x)
    x_max = max(long_stroke.x)
    y_min = min(long_stroke.y)
    y_range = float(y_max - y_min)
    normalized_strokes = []
    for stroke in strokes:
        x = ((np.array(stroke.x) - x_min) / y_range).tolist()
        y = ((np.array(stroke.y) - y_min) / y_range).tolist()
        normalized_strokes.append(Stroke(x, y))
    return normalized_strokes


def group_normalize_wrt_max(strokes):
    """ normailize a multistroke drawing with respect to the x axis """

    long_stroke = concat(strokes)
    x_min = min(long_stroke.x)
    x_max = max(long_stroke.x)
    y_min = min(long_stroke.y)
    x_range = float(x_max - x_min)
    y_range = float(y_max - y_min)
    max_range = max(x_range, y_range)
    normalized_strokes = []
    for stroke in strokes:
        x = ((np.array(stroke.x) - x_min) / max_range).tolist()
        y = ((np.array(stroke.y) - y_min) / max_range).tolist()
        normalized_strokes.append(Stroke(x, y))
    return normalized_strokes


def best_aligment(stroke1, stroke2, indice=None):
    """compare naive euclidian distance, smart euclidian distance 
       and smart euclidian distance after reverting one of the two strokes
       stroke1 and stroke2 must have the same size, otherwize we take the size of the smallest and cut the other"""

    # PATHOLOGIC CASES :
    if min(len(stroke1.x), len(stroke2.x)) == 0:
        return 0, 0, 0, 0, 0, 0

    if indice and indice < len(stroke2.x):
        stroke2 = Stroke(stroke2.x[indice:], stroke2.y[indice:])

    if len(stroke1.x) > len(stroke2.x):
        stroke1 = Stroke(stroke1.x[:len(stroke2.x)], stroke1.y[:len(stroke2.y)])

    if len(stroke2.x) > len(stroke1.x):
        stroke2 = Stroke(stroke2.x[:len(stroke1.x)], stroke2.y[:len(stroke1.y)])

    # ALGORITHM :
    (nx1, ny1, d1, d2, m1, m2) = align(stroke1, stroke2)
    (rx1, ry1, d3, d4, m3, m4) = align(stroke1.revert(), stroke2)

    if np.sum(m4) < np.sum(m2):
        nx1 = rx1
        ny1 = ry1
        d2 = d4
        m2 = m4

    if np.sum(m1) < np.sum(m2):
        nx1 = stroke1.x
        ny1 = stroke2.y
        d2 = d1
        m2 = m1

    if np.sum(m3) < np.sum(m2):
        nx1 = stroke1.revert().x
        ny1 = stroke2.revert().y
        d2 = d3
        m2 = m3

    return nx1, ny1, np.mean(d2), np.mean(m2), d2, m2


def align(stroke1, stroke2):
    """aligne two strokes in order to compute 
       the euclidian distance between them in a smart way"""

    x1 = np.array(stroke1.x)
    x2 = np.array(stroke2.x)
    y1 = np.array(stroke1.y)
    y2 = np.array(stroke2.y)

    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    m = d - np.min(d)

    Ix1 = np.argmax(x1)
    Ix2 = np.argmax(x2)
    Iy1 = np.argmax(y1)
    Iy2 = np.argmax(y2)

    ix1 = np.argmin(x1)
    ix2 = np.argmin(x2)
    iy1 = np.argmin(y1)
    iy2 = np.argmin(y2)

    # rephasing :
    u = np.array([(Ix1 - Ix2), (Iy1 - Iy2), (ix1 - ix2), (iy1 - iy2)])
    indice_period = np.argmin(np.abs(u))
    period = u[indice_period]
    new_x1 = np.array(x1[period:].tolist() + x1[0:period].tolist())
    new_y1 = np.array(y1[period:].tolist() + y1[0:period].tolist())
    x1 = new_x1
    y1 = new_y1

    # resorting : if symetric part, revert it
    mx = np.max((x1, x2), 0)
    my = np.max((y1, y2), 0)
    sym_score = abs(x1 - x2[::-1]) + abs(y1 - y2[::-1])
    if len(x1[sym_score < 50]) > 20:
        x1[sym_score < 40] = x1[sym_score < 40][::-1]
        y1[sym_score < 40] = y1[sym_score < 40][::-1]

    new_d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    new_m = new_d - min(new_d)

    return x1, y1, d, new_d, m, new_m


def euclidian_distance(stroke1, stroke2):
    """the euclidian distance between two strokes
    with same sizes"""

    x1 = np.array(stroke1.x)
    x2 = np.array(stroke2.x)
    y1 = np.array(stroke1.y)
    y2 = np.array(stroke2.y)

    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    m = d - np.min(d)
    if np.mean(m) < 0:
        return 0, 0
    else:
        return np.mean(d), np.mean(m)


def identify(strokes, stroke, closest=True):
    """ look for the best matching postion of a stroke inside a concatenation of a multistroke drawing """

    # better : 1) unifore stroke/stroke ~ relative distance, 2) concatenate

    stroke_length, _ = stroke.euclidian_length()
    stroke.uniformize()
    stroke_num_points = len(stroke.get_x())

    uniformized_strokes = []
    for stroke_i in strokes:
        stroke_i_length, _ = stroke_i.euclidian_length()
        stroke_i.uniformize()
        numDesiredPoints = int(stroke_num_points * float(stroke_i_length) / (float(stroke_length) + 0.0001))
        stroke_i.downsampleShape(numDesiredPoints)
        uniformized_strokes.append(stroke_i)

    draw = concat(uniformized_strokes)
    draw.len = len(draw.x)

    pose = 0
    _, _, best_score, best_match, _, best_values = best_aligment(stroke, draw, pose)

    if closest:
        for i in 1 + np.array(range(draw.get_len() - stroke.get_len() + 1)):
            _, _, score, match, _, values = best_aligment(stroke, draw, i)
            if score < best_score:
                best_score = score
                best_match = match
                best_values = values
                pose = i
    else:
        for i in 1 + np.array(range(draw.get_len() - stroke.get_len() + 1)):
            _, _, score, match, _, values = best_aligment(stroke, draw, i)
            if match < best_match:
                best_score = score
                best_match = match
                best_values = values
                pose = i

    # print best_score

    split_points = draw.split_non_differentiable_points(1.5)

    # plt.plot(draw.x,draw.y,'bo')
    # plt.plot(draw.x[pose:pose+stroke.len],draw.y[pose:pose+stroke.len],'rs')
    # plt.plot(split_points.x,split_points.y,'gs')
    # plt.show()

    if closest:
        return best_score
    else:
        return best_match


def compare(strokes1, strokes2):
    """ takes two multistrokes drawing, alignes them and then compute the euclidian distance """

    score = 0
    for stroke_i in strokes1:
        match = identify(strokes2, stroke_i)
        score += match

    # draw1 = concat(strokes1)
    # draw2 = concat(strokes2)
    # draw1_length,_ = draw1.euclidian_length()
    # draw2_length,_ = draw2.euclidian_length()

    # tot_length = draw1_length# + draw2_length

    return score


def cloud_dist(stroke1, stroke2):
    couples = set()
    distance = 0
    # 1 --> 2
    for i in range(len(stroke1.x)):
        dists_x = np.array(stroke2.x) - stroke1.x[i]
        dists_y = np.array(stroke2.y) - stroke1.y[i]
        dists = np.sqrt(dists_x ** 2 + dists_y ** 2)
        val = np.min(dists)
        j = np.argmin(dists)
        if (i, j) in couples:
            pass
        else:
            distance += val
            couples.add((i, j))
    # 2 --> 1
    for j in range(len(stroke2.x)):
        dists_x = np.array(stroke1.x) - stroke2.x[i]
        dists_y = np.array(stroke1.y) - stroke2.y[i]
        dists = np.sqrt(dists_x ** 2 + dists_y ** 2)
        val = np.min(dists)
        i = np.argmin(dists)
        if (i, j) in couples:
            pass
        else:
            distance += val
            couples.add((i, j))

    return distance / float(len(couples))
