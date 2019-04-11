#!/usr/bin/env python
# coding: utf-8

"""
Hand-written letters
"""

import math
import copy
import pickle
import numpy as np
from scipy import interpolate
from stroke import Stroke

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

def merge_words(words_list, fast_mode = True):
    """
    Combines all the words using the letter merging algorithm where
    words_list = [[p1_l1, p1_l2], [p2_l1, p2_l2], [p3_l1, p3_l2])
    and pi_lj is the demonstration written by person i for letter j
    Assumes the list contains demonstrations by at least one participant
    """
    n_participants, n_letters = len(words_list), len(words_list[0])
    letter_demonstrations = [[words_list[i][j] for i in range(n_participants)] for j in range(n_letters)]
    letter_properties =  [[words_list[i][j].original_dimensions() for i in range(n_participants)] for j in range(n_letters)]

    # Merge the different demonstration letters provided
    merged_letters = [safe_merge_letters(x, fast_mode) for x in letter_demonstrations]

    # Reconstruct the word from the individual letters by rescaling and
    # translating them based on the original letter properties
    rescaled_letters = copy.deepcopy(merged_letters)
    for letter, properties in zip(rescaled_letters, letter_properties):
        loffset_x = max([ p["xmin"] for p in properties]) # max of the min_x
        loffset_y = max([ p["ymin"] for p in properties]) # max of the min_y
        lscale_x = np.array([ p["width"] for p in properties]).mean() # avg width
        lscale_y = np.array([ p["height"] for p in properties]).mean() # avg height
        letter.scale_and_translate(s_x = lscale_x, s_y = lscale_y, t_x = loffset_x, t_y = loffset_y)

    # Return the reconstructed word and individual letters that are normalized
    return rescaled_letters, merged_letters

def safe_merge_letters(letters_list, fast_mode = True):
    try:
        return merge_letters(letters_list, fast_mode)
    except:
        if fast_mode:
            print "Error in letter merging, reverting to normal mode"
            return merge_letters(letters_list, fast_mode = False)
        else:
            print "Error merging the given letters"

def merge_letters(letters_list, fast_mode = True):
    """
    Combines all the letter class objects based on a stroke by stroke process
    """
    # Input : all the demonstrations of a letter in the form of a list of Letter objects
    preprocessed_letters = copy.deepcopy(letters_list)

    letter_properties = dict()
    for idx, letter in enumerate(preprocessed_letters):
        letter.normalize_wrt_max()
        if not fast_mode:
            letter.uniformize_strokes_with_step(step=0.01)
            letter.combine_strokes_if_low_density(density_limit=10)
        letter.uniformize_strokes_with_step(step=0.1)

    n_strokes = min([letter.n_strokes() for letter in preprocessed_letters])

    for letter in preprocessed_letters:
        letter.reduce_to_n_strokes(n_strokes)

    n_pts_per_stroke = [min([letter.stroke_len(idx_stroke) for letter in preprocessed_letters]) for idx_stroke in range(n_strokes)]
    n_pts_per_stroke = [min(100, n_pts) for n_pts in n_pts_per_stroke]

    for idx, letter in enumerate(preprocessed_letters):
        letter.uniformize_strokes(n_pts_per_stroke = n_pts_per_stroke)

    combined_strokes_x, combined_strokes_y = [], []

    for i in range(n_strokes):
        demonstration_strokes = [letter.stroke(i) for letter in preprocessed_letters]
        # Stroke by stroke weighted sum - Assumes all the strokes are of equal length
        x_list = [np.array(stroke.get_x()) for stroke in demonstration_strokes]
        y_list = [np.array(stroke.get_y()) for stroke in demonstration_strokes]

        n_demonstrations = len(demonstration_strokes)
        x = sum(x_list)/n_demonstrations
        y = sum(y_list)/n_demonstrations

        combined_strokes_x.append(x)
        combined_strokes_y.append(y)

    combined_letter = Letter(combined_strokes_x, combined_strokes_y)
    combined_letter.normalize_wrt_max()

    return combined_letter

def dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


class Letter:
    """ a letter object is a collection of strokes """

    def __init__(self, x, y):
        """
        Provided only x an y, they are expected to be a list of list of coordinates
        e.g. x = [[0,1,2], [3,4,5], [6,7,8]], y = [[9,10,11], [12,13,14], [15,16,17]],
             pen_up = None
        Provided x, y and pen_up, x and y are expected to be a list of coordinates
        and pen_up a list of 0 and 1 where 1 means the pen was removed from the
        tablet at the considered index
        e.g. x = [0,1,2,3,4,5,6,7,8], y = [9,10,11,12,13,14,15,16,17]
        """
        self.x_list, self.y_list = x, y

        self.x_list = [x for x in self.x_list if len(x)>0]
        self.y_list = [x for x in self.y_list if len(x)>0]

        self.strokes = [Stroke(x_sublist, y_sublist) for x_sublist, y_sublist in zip(self.x_list, self.y_list)
                        if len(x_sublist)>0 and len(y_sublist)>0]

        self.original_xmin = min([min(sublist) for sublist in self.x_list]) if(len(self.strokes)>0) else 0
        self.original_xmax = max([max(sublist) for sublist in self.x_list]) if(len(self.strokes)>0) else 0
        self.original_ymin = min([min(sublist) for sublist in self.y_list]) if(len(self.strokes)>0) else 0
        self.original_ymax = max([max(sublist) for sublist in self.y_list]) if(len(self.strokes)>0) else 0
        self.original_width = (self.original_xmax - self.original_xmin) if(len(self.strokes)>0) else 0
        self.original_height = (self.original_ymax - self.original_ymin) if(len(self.strokes)>0) else 0

        self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height = 0, 0, 0, 0, 0, 0
        self.compute_letter_properties()

    def compute_letter_properties(self):
        if(len(self.strokes)>0):
            self.xmin = min([min(stroke.get_x()) for stroke in self.strokes if len(stroke.get_x())>0])
            self.xmax = max([max(stroke.get_x()) for stroke in self.strokes if len(stroke.get_x())>0])
            self.ymin = min([min(stroke.get_y()) for stroke in self.strokes if len(stroke.get_y())>0])
            self.ymax = max([max(stroke.get_y()) for stroke in self.strokes if len(stroke.get_y())>0])
            self.width = self.xmax-self.xmin
            self.height = self.ymax-self.ymin

    def strokes_idx(self, pen_up):
        idx = [i for i in xrange(len(pen_up)) if pen_up[i]==1]
        idx += [len(pen_up)-1]
        return idx

    def orig_x(self):
        return self.x

    def orig_y(self):
        return self.y

    def x(self):
        return [stroke.get_x() for stroke in self.strokes]

    def y(self):
        return [stroke.get_y() for stroke in self.strokes]

    def stroke_len(self, idx_stroke):
        return len(self.strokes[idx_stroke].get_x())

    def get_strokes(self):
        return self.strokes

    def to_stroke_with_pen_up(self):
        x = sum([stroke.get_x() for stroke in self.strokes], [])
        y = sum([stroke.get_y() for stroke in self.strokes], [])

        pen_ups = [[0 for i in range(len(stroke.get_x()))] for stroke in self.strokes]
        for pen_up in pen_ups:
            pen_up[0] = 1
        pen_ups = sum(pen_ups, [])

        return Stroke(x,y), pen_ups

    def original_dimensions(self):
        return {"xmin":self.original_xmin, "xmax":self.original_xmax,
                "ymin":self.original_ymin, "ymax":self.original_ymax,
                "width": self.original_width, "height":self.original_height}

    def current_dimensions(self):
        return {"xmin":self.xmin, "xmax":self.xmax, "ymin":self.ymin, "ymax":self.ymax,
                "width": self.width, "height":self.height}

    def n_strokes(self):
        return len(self.strokes)

    def stroke(self, idx_stroke):
        if idx_stroke<len(self.strokes):
            return self.strokes[idx_stroke]
        else:
            print("Invalid stroke id ({}), max id is {}".format(idx_stroke, len(self.strokes)))
            print("Returning empty stroke")
            return Stroke()

    def reduce_to_n_strokes(self, n_strokes):
        """
        Combines successive strokes based on minimum distance criteria
        until reaching n_strokes, the desired number of strokes
        """
        if n_strokes == len(self.strokes):
            return
        elif n_strokes <= 0:
            print("Invalid number of strokes")
        elif n_strokes>len(self.strokes):
            print("Cannot reduce to {} strokes as {} > to the number of strokes ({})".format(n,n,len(self.strokes)))
        else:
            #Distance between the end point of a stroke and the start of the following one
            dists = [dist(s1.get_x()[-1],s1.get_y()[-1],s2.get_x()[0],s2.get_y()[0])
                     for (s1, s2) in zip(self.strokes[:-1], self.strokes[1:])]
            n_strokes_to_remove = len(self.strokes)-n_strokes
            for i in range(n_strokes_to_remove):
                idx_min  = dists.index(min(dists))
                x_pts = list(self.strokes[idx_min].get_x())+list(self.strokes[idx_min+1].get_x())
                y_pts = list(self.strokes[idx_min].get_y())+list(self.strokes[idx_min+1].get_y())
                self.strokes[idx_min] = Stroke(x_pts, y_pts)# Put the combination at min index
                self.strokes.remove(self.strokes[idx_min+1])# Remove the second part of the new stroke from letter list
                dists.remove(min(dists)) # Recompute the dists

    def uniformize_strokes_with_step(self, step = 0.1):
        strokes = list()
        for stroke in self.strokes:
            if(len(stroke.get_x())<=1 and len(stroke.get_y())<=1):
                continue
            x_pts, y_pts = stroke.get_x(), stroke.get_y()
            densified_stroke = [evenly_spaced_interpolation(x1,y1,x2,y2) for x1, y1, x2, y2
                                        in zip(x_pts[:-1], y_pts[:-1], x_pts[1:], y_pts[1:])]
            x, y = [s["x"] for s in densified_stroke], [s["y"] for s in densified_stroke]
            x, y = sum(x, []), sum(y, [])
            strokes.append(Stroke(x,y))
        self.strokes = strokes

    def uniformize_strokes(self, n_pts_per_stroke):
        strokes = []
        for stroke, n_pts in zip(self.strokes, n_pts_per_stroke):
            stroke.downsampleShape(n_pts) # Downsample to have the same number of points
            stroke.uniformize() # Get equidistant points along the stroke
            strokes.append(stroke)

        self.strokes = strokes

    def combine_strokes_if_low_density(self, density_limit=10):
        all_x, all_y = [l.get_x() for l in self.strokes], [l.get_y() for l in self.strokes]
        all_x, all_y = sum(all_x, []), sum(all_y, [])
        full_letter = Stroke(all_x, all_y)
        full_pen_up = [compute_pen_up(letter.get_x(), letter.get_y(), density_limit = 10) for letter in self.strokes]
        full_pen_up = sum(full_pen_up, [])
        new_strokes_idx = self.strokes_idx(full_pen_up)
        x_pts = [full_letter.get_x()[idx_start:idx_end] for (idx_start, idx_end)
                            in zip(new_strokes_idx[:-1], new_strokes_idx[1:])]
        y_pts = [full_letter.get_y()[idx_start:idx_end] for (idx_start, idx_end)
                            in zip(new_strokes_idx[:-1], new_strokes_idx[1:])]
        self.strokes = [Stroke(x,y) for x,y in zip(x_pts, y_pts)]
        self.uniformize_strokes_with_step(step = 0.1)
        self.compute_letter_properties()

    def normalize_wrt_max(self):
        """
        Normalizes all the strokes making up the letter
        """
        #Normalize at the letter scale
        self.compute_letter_properties()
        max_range = max(self.width, self.height)

        for stroke in self.strokes:
            x = (np.array(stroke.get_x())-self.xmin)/float(max_range)
            y = (np.array(stroke.get_y())-self.ymin)/float(max_range)
            stroke.x = x.tolist()
            stroke.y = y.tolist()

        self.compute_letter_properties()

    def scale_and_translate(self, s_x = 1, s_y = 1, t_x = 0, t_y = 0):
        """
        Multiplies the coordinates by s then translates by t
        Assumes normalization was done beforehand
        """

        for stroke in self.strokes:
            x = np.array(stroke.get_x())*s_x+t_x
            y = np.array(stroke.get_y())*s_y+t_y
            stroke.x = x.tolist()
            stroke.y = y.tolist()

        self.compute_letter_properties()

def remove_redundant_points(x_pts, y_pts):
    dists = [dist(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(x_pts[:-1], y_pts[:-1], x_pts[1:], y_pts[1:])]
    same_idx = [i for i in xrange(len(dists)) if dists[i] == 0]
    x = [x_pts[i] for i in xrange(len(x_pts)) if i not in same_idx]
    y = [y_pts[i] for i in xrange(len(y_pts)) if i not in  same_idx]
    return x, y

def evenly_spaced_interpolation(x1,y1,x2,y2, step = 0.001):
    dx, dy = x2-x1, y2-y1
    theta = math.atan2(dy, dx)
    dist = np.sqrt(dx**2+dy**2)

    if dist<step:
        x = [x1,x2]
        y = [y1,y2]
    else:
        n_pts = int(np.round(dist/step))+1
        new_step = dist/(n_pts-1)
        x_pts = [x1+i*new_step*math.cos(theta) for i in xrange(n_pts)]
        y_pts = [y1+i*new_step*math.sin(theta) for i in xrange(n_pts)]
        x, y = remove_redundant_points(x_pts, y_pts)

    return {"x":x, "y":y}

def compute_pen_up(x_pts, y_pts, density_limit = 3):
    if len(x_pts)>1:
        # Artificially set the pen up variable in the merged variant of the word
        dists = [dist(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(x_pts[:-1], y_pts[:-1], x_pts[1:], y_pts[1:])]
        mean_dist = np.mean(dists)
        density = np.array(dists)/mean_dist
        pen_ups = np.zeros(len(density))
        pen_ups = [1 if d > density_limit else 0 for d in density]
        pen_ups = [1] + pen_ups
        pen_ups[-1] = 0
    else:
        pen_ups = [1]
    return pen_ups

cmaps = ["cool", "autumn", "summer", "spring", "winter"]
def plot_words(words_list, color_mode = "black", title = "", display_idx = False):
    fig=plt.figure(figsize=(10,10))
    plt.title(title)
    gs=GridSpec(len(words_list), max([len(word) for word in words_list]))
    ax_words = [fig.add_subplot(gs[i,:]) for i in range(len(words_list))]

    for ax, word in zip(ax_words, words_list):
        idx_color = 0
        for letter in word:
            for idx, stroke in enumerate(letter.get_strokes()):
                x = stroke.get_x()
                y = -np.array(stroke.get_y())
                if color_mode == "black":
                    ax.plot(x, y, 'k')
                elif color_mode == "rainbow":
                    ax.plot(x, y)
                elif "gradient" in color_mode:
                    cmap = cmaps[idx_color%len(cmaps)] if "rainbow" in color_mode else "viridis"
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    dydx = np.array([i for i in range(len(x))])

                    # Create a continuous norm to map from data points to colors
                    norm = plt.Normalize(dydx.min(), dydx.max())
                    lc = LineCollection(segments, cmap=cmap, norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(dydx)
                    lc.set_linewidth(2)
                    line = ax.add_collection(lc)

                idx_color = idx_color+1

                if display_idx:
                    bbox_props = dict(boxstyle="circle,pad=0.3", fc="k", lw=2)
                    ax.annotate(idx+1,
                        xy=(x[0], y[0]),
                        xytext=(x[0], y[0]),
                        color='w', size = 20,
                        bbox=bbox_props)

        ax.axis("equal")
        plt.axis('off')
    #plt.show()

if __name__=="__main__":
    data = pickle.load( open( "merged_strokes_3_tipfjklx.pickle", "rb" ) )
    words_list = []

    for participant in data.keys():
        p_letters, p_pen_ups = data[participant]["strokes"], data[participant]["pen_ups"]
        participant_letters = []

        for letter, pen_up in zip(p_letters, p_pen_ups):
            x_strokes, y_strokes = [], []
            idx = [i for i in xrange(len(pen_up)) if pen_up[i]==1]
            idx += [len(pen_up)-1]
            x, y = letter.get_x(), letter.get_y()
            x_pts = [x[idx_start:idx_end+1] for (idx_start, idx_end)
                                in zip(idx[:-1], idx[1:])]
            y_pts = [y[idx_start:idx_end+1] for (idx_start, idx_end)
                                in zip(idx[:-1], idx[1:])]
            for stroke_x, stroke_y in zip(x_pts, y_pts):
                x_strokes.append(stroke_x)
                y_strokes.append(stroke_y)

            letter = Letter(x_strokes, y_strokes)
            participant_letters.append(letter)
        words_list.append(participant_letters)

    rescaled_letters, merged_letters = merge_words(words_list)

    for color_mode in ["black", "gradient", "rainbow", "rainbow_gradient"]:
        plot_words(words_list, title="original words_{}".format(color_mode), color_mode = color_mode)
        plot_words([rescaled_letters], title="merged_words_{}".format(color_mode), color_mode = color_mode)
    plt.show()
