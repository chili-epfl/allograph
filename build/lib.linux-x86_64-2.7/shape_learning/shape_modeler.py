""" Class to decompose a dataset of shapes into its principle components,
and to make and show new shapes which are represented by the mean shape
plus some amount of said principle components.
"""

import random
from copy import deepcopy

import numpy
import os.path

import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

class ShapeModeler:
    def __init__(self,
                 shape_name=None, 
                 samples=None, 
                 init_filename=None,
                 update_filenames=None,
                 param_filename=None,
                 robot_filenames=None,
                 num_principle_components=10):
        """ Initialize a shape modeler

        If given an initial dataset (params samples or init_filename), loads the training
        dataset for a given shape, and run a PCA decomposition on it.
        
        The update dataset (a file name or a list of file names) are used to serialize 
        the shapes from demonstrations. If it's a list, the first file will contains
        the initial shapes + the demo shapes, and the followings will contain just
        the demo shapes.

        :param samples: a list of sample shapes, each presented as a list of coordinates [x1,x2,...,y1,y2...].
                        Each shape must have the same number of coordinates.
        :param filename: the path to a shape dataset. See makeDataMatrix for the expected format.
        :paran num_principle_components: the number of desired principle components
        """

        self.shape_name = shape_name
        self.num_principle_components = num_principle_components

        if samples is None and init_filename is None:
            return

        if samples:
            self.dataMat = numpy.array(samples)

        elif init_filename:
            self.makeDataMatrix(init_filename)

        if update_filenames:
            self.update_filenames = update_filenames

        if param_filename:
            self.param_filename = param_filename

        if robot_filenames:
            self.robot_filenames = robot_filenames

        if not isinstance(update_filenames,list):
                self.update_filenames = [update_filenames]

        self.performPCA()
        
        (self.refParams, error) = self.decomposeShape(numpy.reshape(self.dataMat[0], (-1, 1)))
        #self.createNewSet()

    def makeDataMatrix(self, filename):
        """Read data from text file and store resulting data matrix

        For n samples of m points, the text file should be formatted as:

            n
            m
            x11 x12 ... x1m y11 y12 ... y1m
            x21 x22 ... x2m y21 y22 ... y2m
            ...
            xn1 xn2 ... xnm yn1 yn2 ... ynm

        """

        # scan the dataset :
        lines = []
        try: 
            with open(filename, 'r') as f:
                lines.append(f.readline())
                lines.append(f.readline())
                nb_samples =  int(lines[1].strip())
                for i in range(nb_samples+5):
                    lines.append(f.readline())
        except IOError:
            raise RuntimeError("no reading permission for file"+filename)

        self.numShapesInDataset = int(lines[1].strip())
        self.numPointsInShapes = int(lines[3].strip())
        if not (self.numShapesInDataset and self.numPointsInShapes):
            raise RuntimeError("Unable to read sizes needed from text file")

        self.numShapesInDemo = 0
        self.dataMat = numpy.empty((self.numShapesInDataset, self.numPointsInShapes * 2))
        self.demoDataMat = numpy.empty((self.numShapesInDemo, self.numPointsInShapes * 2))
        
        ref_line = lines[5].strip()
        ref_values =  ref_line.split(' ')
        if not (len(ref_values) == self.numPointsInShapes * 2):
            raise RuntimeError("Unable to read appropriate number of points from text file for reference shape ")
        self.refShape = map(float, ref_values)
        self.dataMat[0] = self.refShape

        for i in range(self.numShapesInDataset-1): # -1 because we have already add the first shape (reference)
            line = lines[i+7].strip()
            values = line.split(' ')
            if not (len(values) == self.numPointsInShapes * 2):
                raise RuntimeError(
                    "Unable to read appropriate number of points from text file for shape " + str(i + 1))
            self.dataMat[i+1] = map(float, values)

    def readStartingPoint(self):
        """"""
        # start_shape_line = ''
        # open the starting-point :
        try:
            with open(self.robot_filenames[0], 'r') as f:
                start_shape_line = f.readlines()[-1]
                print self.robot_filenames[0]
                print start_shape_line
        except IOError:
            raise RuntimeError("no reading permission for file"+self.robot_filenames[0])

        start_shape_values = start_shape_line.split(' ')
        try:
            start_shape_values.remove('\n')
        except ValueError:
            pass
        while True:
            try:
                start_shape_values.remove('')
            except ValueError:
                break
        if not (len(start_shape_values) == self.numPointsInShapes * 2):
            print start_shape_values
            print len(start_shape_values)
            print self.numPointsInShapes * 2
            raise RuntimeError("Unappropriate starting shape format")
        self.startShape = numpy.array(map(float, start_shape_values))
        self.startShape = numpy.reshape(self.startShape, (-1, 1))
        return self.startShape


    def performPCA(self):
        """ Calculate the top 'num_principle_components' principle components of
        the dataset, the observed variance of each component, and the mean
        """
        covarMat = numpy.cov(self.dataMat.T)
        eigVals, eigVecs = numpy.linalg.eig(covarMat)
        self.principleComponents = numpy.real(eigVecs[:, 0:self.num_principle_components])
        self.parameterVariances = numpy.real(eigVals[0:self.num_principle_components])
        self.meanShape = self.dataMat.mean(0).reshape((self.numPointsInShapes * 2, 1))

    def getEuclidianCenter(self):
        """ Get the euclidian mean point by point
        return a vector of 70 mean-points"""
        return numpy.mean(self.dataMat,0),numpy.var(self.dataMat,0)

    def getEuclidianDist(self, shape):
        """ Get the euclidian distance between a shape and the mean shape
        of the dataset, point by point, divided by variance for normalization """
        #data_mean, data_var = self.getEuclidianCenter()
        dist = self.meanShape - shape
        #sigma =numpy.sqrt(self.parameterVariances)
        return numpy.sum(dist*dist)

    def getDistToRef(self,shape):
        params1,_ = self.decomposeShape(self.dataMat[0].reshape((self.numPointsInShapes * 2, 1)))
        params2,_ = self.decomposeShape(shape)
        dist = numpy.array(params1)-numpy.array(params2)
        var = numpy.abs(numpy.array(self.getParameterVariances()))
        ndist = dist*dist/var
        return numpy.sum(ndist[:1])

    def getReference(self):
        """ gives the refence shape of the letter """
        return self.dataMat[0]

    def getParameterVariances(self):
        """ Return the variances associated which each of the top principle components
        """
        return self.parameterVariances

    def makeShape(self, params):
        """ Generate a shape with the given parameter vector
        """
        if (not params.shape == (self.num_principle_components, 1)):
            raise RuntimeError("Vector of parameters must have dimensions of (num_principle_components,1)")
        shape = self.meanShape + numpy.dot(self.principleComponents, params)
        return shape

    def makeShapeVaryingParam(self, paramsToVary, paramValues):
        """ Generate a shape modifying the given parameter
        """
        xb = numpy.zeros((self.num_principle_components, 1))
        for i in range(len(paramsToVary)):
            xb[paramsToVary[i] - 1, 0] = paramValues[i]
        shape = self.makeShape(xb)
        return shape, xb

    def makeRandomShapeFromUniform(self, params, paramsToVary, bounds):
        """ Draw 'paramsToVary' values from uniform distribution with limits
        given by 'bounds' and make shape
        """
        xb = deepcopy(params)
        for i in range(len(paramsToVary)):
            sample = random.uniform(bounds[i, 0], bounds[i, 1])
            xb[paramsToVary[i] - 1, 0] = sample
        shape = self.makeShape(xb)
        return shape, xb

    def makeRandomShapeFromTriangular(self, params, paramsToVary, bounds, modes):
        """ Draw 'paramsToVary' values from triangular distribution with limits
        given by 'bounds' and modes given by 'modes' and make shape       
        """
        b = deepcopy(params)
        for i in range(len(paramsToVary)):
            sample = random.triangular(bounds[i, 0], modes[i], bounds[i, 1])
            b[paramsToVary[i] - 1, 0] = sample
        return self.makeShape(b), b

    def decomposeShape(self, shape):
        """ Convert shape into its 'num_principle_components' parameter values
        (project it onto the num_principle_components-dimensional space)
        """
        if (not shape.shape == (self.numPointsInShapes * 2, 1)):
            print(shape.shape)
            print(self.numPointsInShapes)
            raise RuntimeError("Shape to decompose must be the same size as shapes used to make the dataset")
        params = numpy.dot(self.principleComponents.T, shape - self.meanShape)

        approxShape = self.meanShape + numpy.dot(self.principleComponents, params)
        diff = abs(shape - approxShape) ** 2
        error = sum(diff) / (self.numPointsInShapes * 2)
        return params, error

    def normaliseMeanShapeHeight(self):
        self.meanShape = ShapeModeler.normaliseShapeHeight(self.meanShape)

    def showMeanShape(self, block=True):
        ShapeModeler.showShape(ShapeModeler.normaliseShape(self.meanShape), block)
        
        
    def extendDataMat(self, shape):
        """ Add the demonstrated shape to the data matrix to performe PCA. 
        We want to update principle components by taking into account the demo shapes.
        """
        self.numShapesInDataset+=1
        self.numShapesInDemo+=1
        self.dataMat = numpy.append(self.dataMat, shape.T, axis=0)
        self.demoDataMat = numpy.append(self.demoDataMat, shape.T, axis=0)
        self.performPCA()
        (self.refParams, error) = self.decomposeShape(numpy.reshape(self.dataMat[0], (-1, 1)))

    def save_all(self):
        """ save the inital shape + the demo shapes into a new dataset.
        """
        if self.update_filenames:
            filename = self.update_filenames[0]
            #print('saving in'+filename)
            if not os.path.exists(filename):
                raise RuntimeError("path to dataset"+filename+"not found")
            try:
                with open(filename, 'wb') as f:
                    f.write('nb_sample:\n')
                    f.write('%i\n'%self.numShapesInDataset)
                    f.write('nb_pts:\n')
                    f.write('%i\n'%self.numPointsInShapes)
                    f.write('ref:\n')
                    f.write(' '.join(map(str,self.dataMat[0])) + '\n')
                    f.write('...\n')
                    for i in range(len(self.dataMat)-1):
                        f.write(' '.join(map(str,self.dataMat[i+1])))
                        f.write('\n')
            except IOError:
                raise RuntimeError("no writing permission for file"+filename)

    def save_demo(self):
        """ save the demo shape into a new data set.
        """
        if len(self.update_filenames)>1:
            for filename in self.update_filenames[1:]:
                if not os.path.exists(filename):
                    raise RuntimeError("path to dataset"+filename+"not found")
                try:
                    with open(filename, 'wb') as f:
                        f.write('nb_sample:\n')
                        f.write('%i\n'%self.numShapesInDataset)
                        f.write('nb_pts:\n')
                        f.write('%i\n'%self.numPointsInShapes)
                        f.write('ref:\n')
                        f.write(' '.join(map(str,self.dataMat[0])) + '\n')
                        f.write('...\n')
                        for i in range(len(self.dataMat)-1):
                            f.write(' '.join(map(str,self.demoDataMat[i+1])))
                            f.write('\n')
                except IOError:
                    raise RuntimeError("no writing permission for file"+filename)

    def save_robot_try(self,last_try):
        """ save the curent learned shape of the robot into a new data set.
        """
        if self.robot_filenames:
            filename = self.robot_filenames[0]
            #print('saving robot try in'+filename)
            if not os.path.exists(filename):
                raise RuntimeError("path to dataset"+filename+"not found")
            try:
                with open(filename, "a") as f:
                    f.write(' '.join(map(str,last_try)).replace('[','').replace(']','').replace('\n',''))
                    f.write('\n')
            except IOError:
                raise RuntimeError("no writing permission for file"+filename)

    def save_params(self, params, letter):
        """save parameters in new dataset
        """
        if self.param_filename:
            filename = self.param_filename
            print('saving params in'+filename)
            if not os.path.exists(filename):
                raise RuntimeError("path to dataset"+filename+"not found")
            lines = []
            try:
                with open(filename, 'rb') as f:
                    for i in range(52):
                        lines.append(f.readline())
            except IOError:
                raise RuntimeError("no reading permission for file"+filename)
            try:
                with open(filename, 'wb') as f:
                    for i in range(52):
                        f.write(lines[i])
                        test = lines[i].replace('[','').replace(']\n','')==letter
                        if test:
                            lines[i+1] = str(params).replace('[','').replace(']','')+'\n'
            except IOError:
                raise RuntimeError("no writing permission for file"+filename)


    def paramMatrix(self):
        paramMat = numpy.zeros((self.numShapesInDataset,self.num_principle_components))
        for i in range(self.numShapesInDataset):
            shape = self.dataMat[i,:]
            params,_ = self.decomposeShape(shape.reshape((self.numPointsInShapes * 2, 1)))
            paramMat[i,:] = params[:,0]
        return paramMat

    def getVar(self):
        return numpy.var(self.dataMat,0)

    def getClusters(self):
        """
        get the different clusters of the letter
        """
        X = self.dataMat

        ms = MeanShift(bandwidth=1.9).fit(X)
        cluster_centers = ms.cluster_centers_
        n_clusters = len(cluster_centers)
        labels = ms.labels_
        var = []
        for i in range(n_clusters):
            var.append(numpy.var(X[labels==i]))
        var = numpy.array(var)

        return cluster_centers, n_clusters, var

    def getMinDist(self,shape):
        """give the distance between the demo and the closest cluster of the letter"""

        clusters,_,var = self.getClusters()
        scores = []
        for i in range(len(clusters)):
            dist = (clusters[i,:].reshape((self.numPointsInShapes * 2, 1))-shape)
            scores.append(numpy.sum(dist*dist))
        return numpy.min(numpy.array(scores))

    def getCenters(self):

        clusters,_,_ = self.getClusters()
        centers = [clusters[i,:].reshape((self.numPointsInShapes * 2, 1)) for i in range(len(clusters))]
        return centers



    @staticmethod
    def showShape(shape, block=False):
        """ Show shape with random colour
        """
        numPointsInShape = len(shape) / 2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        plt.plot(x_shape, -y_shape, c=numpy.random.rand(3, 1))
        plt.axis([-1, 1, -1, 1])
        if block:
            plt.show(block=block)  # block=False <-> plt.draw
        else:
            plt.draw()

    @staticmethod
    def normaliseShape(shape):
        """ Normalise shape so that max dimension is 1 
        """
        numPointsInShape = len(shape) / 2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        # shift so centre of shape is at (0,0)
        x_range = max(x_shape) - min(x_shape)
        y_range = max(y_shape) - min(y_shape)
        x_shape = x_shape - (max(x_shape) - x_range / 2)
        y_shape = y_shape - (max(y_shape) - y_range / 2)

        #normalise shape
        scale = max(x_range, y_range)
        if scale < 1e-10:
            print('Warning: shape is probably a bunch of points on top of each other...')

        x_shape = x_shape / scale
        y_shape = y_shape / scale

        newShape = numpy.zeros(shape.shape)
        newShape[0:numPointsInShape] = x_shape
        newShape[numPointsInShape:] = y_shape
        return newShape

    @staticmethod
    def getShapeCentre(shape):
        """ Calculate the centre of the shape
        """
        numPointsInShape = len(shape) / 2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        x_range = max(x_shape) - min(x_shape)
        y_range = max(y_shape) - min(y_shape)
        x_centre = (max(x_shape) - x_range / 2)
        y_centre = (max(y_shape) - y_range / 2)
        return [x_centre, -y_centre]

    @staticmethod
    def normaliseShapeHeight(shape):
        """ Normalise shape so that height is 1 
        """
        numPointsInShape = len(shape) / 2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        # shift so centre of shape is at (0,0)
        x_range = max(x_shape) - min(x_shape)
        y_range = max(y_shape) - min(y_shape)
        x_centre = (max(x_shape) - x_range / 2)
        y_centre = (max(y_shape) - y_range / 2)
        x_shape = x_shape - x_centre
        y_shape = y_shape - y_centre

        #normalise shape
        scale = y_range
        if scale < 1e-10:
            print('Warning: shape is probably a bunch of points on top of each other...')

        x_shape = x_shape / scale
        y_shape = y_shape / scale

        newShape = numpy.zeros(shape.shape)
        newShape[0:numPointsInShape] = x_shape
        newShape[numPointsInShape:] = y_shape
        return newShape

    @staticmethod
    def normaliseShapeWidth(shape):
        """ Normalise shape so that width is 1 
        """
        numPointsInShape = len(shape) / 2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        # shift so centre of shape is at (0,0)
        x_range = max(x_shape) - min(x_shape)
        y_range = max(y_shape) - min(y_shape)
        x_centre = (max(x_shape) - x_range / 2)
        y_centre = (max(y_shape) - y_range / 2)
        x_shape = x_shape - x_centre
        y_shape = y_shape - y_centre

        #normalise shape
        scale = x_range
        if scale < 1e-10:
            print('Warning: shape is probably a bunch of points on top of each other...')

        x_shape = x_shape / scale
        y_shape = y_shape / scale

        newShape = numpy.zeros(shape.shape)
        newShape[0:numPointsInShape] = x_shape
        newShape[numPointsInShape:] = y_shape
        return newShape

    @staticmethod
    def normaliseAndShowShape(shape, block=False):
        """ Normalise shape so that max dimension is 1 and then show
        """
        shape = ShapeModeler.normaliseShape(shape)
        ShapeModeler.showShape(shape, block)

    @staticmethod
    def showShape_score(shape, scores, block=False):
        """ Show shape with random colour
        """
        numPointsInShape = len(shape) / 2
        x_shape = numpy.reshape(numpy.array(shape[0:numPointsInShape]),(numPointsInShape))
        y_shape = numpy.reshape(numpy.array(shape[numPointsInShape:]),(numPointsInShape))

        #plt.plot(x_shape, -y_shape, c=numpy.random.rand(3, 1))
        #plt.axis([-1, 1, -1, 1])
        if block:
            plt.show(block=block)  # block=False <-> plt.draw
        else:
            plt.errorbar(x_shape,-y_shape, yerr=scores/numpy.max(scores))
            plt.draw()
