#!/usr/bin/env python

"""
Manages a collection of shape_learners, with long-term memory about the 
history of previous collections seen. An example is managing shape_learners
which represent letters, and the collections represent words. 
"""

import logging; shapeLogger = logging.getLogger("shape_logger")
import os.path

from shape_learner import ShapeLearner
from recordtype import recordtype  # for mutable namedtuple (dict might also work)


boundExpandingAmount = 0.
usePrevParamsWhenShapeReappears = True

Shape = recordtype('Shape', [('path', None), ('shapeID', None), ('shapeType', None), ('shapeType_code', None),
                             ('paramsToVary', None), ('paramValues', None)])

def configure_logging(path):

    if path:
        if os.path.isdir(path):
            path = os.path.join(path, "shapes.log")
        handler = logging.FileHandler(path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
    else:
        handler = logging.NullHandler()

    shapeLogger.addHandler(handler)
    shapeLogger.setLevel(logging.DEBUG)


# ##--------------------------------------------- WORD LEARNING FUNCTIONS
class ShapeLearnerManager:
    def __init__(self, generateSettingsFunction, shapes_logging_path = "shapes.log"):

        configure_logging(shapes_logging_path)
        shapeLogger.info("**************** NEW SESSION ***************")

        self.generateSettings = generateSettingsFunction
        self.shapesLearnt = []
        self.shapeLearners_all = []
        self.shapeLearners_currentCollection = []
        self.settings_shapeLearners_all = []
        self.settings_shapeLearners_currentCollection = []
        self.shapeLearnersSeenBefore_currentCollection = []
        self.currentCollection = ""
        self.collectionsLearnt = []
        self.nextShapeLearnerToBeStarted = 0
        self.currentDemo = ""
        self.currentLearn = ""

    def initialiseShapeLearners(self):
        self.shapeLearners_currentCollection = []
        self.settings_shapeLearners_currentCollection = []
        self.shapeLearnersSeenBefore_currentCollection = []
        for i in range(len(self.currentCollection)):
            shapeType = self.currentCollection[i]

            #check if shape has been learnt before
            try:
                shapeType_index = self.shapesLearnt.index(shapeType)
                newShape = False
            except ValueError:
                newShape = True
            self.shapeLearnersSeenBefore_currentCollection.append(not newShape)
            if (newShape):
                settings = self.generateSettings(shapeType)

                shapeLearner = ShapeLearner(settings)
                self.shapesLearnt.append(shapeType)
                self.shapeLearners_all.append(shapeLearner)
                self.settings_shapeLearners_all.append(settings)
                self.shapeLearners_currentCollection.append(self.shapeLearners_all[-1])
                self.settings_shapeLearners_currentCollection.append(self.settings_shapeLearners_all[-1])

            else:
                #use the bounds determined last time
                previousBounds = self.shapeLearners_all[shapeType_index].getParameterBounds()
                newInitialBounds = previousBounds
                newInitialBounds[0, 0] -= boundExpandingAmount;  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
                newInitialBounds[0, 1] += boundExpandingAmount;  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
                self.shapeLearners_all[shapeType_index].setParameterBounds(newInitialBounds)
                self.shapeLearners_currentCollection.append(self.shapeLearners_all[shapeType_index])
                self.settings_shapeLearners_currentCollection.append(self.settings_shapeLearners_all[shapeType_index])


    def startNextShapeLearner(self):
        #start learning
        if ( self.nextShapeLearnerToBeStarted < len(self.currentCollection) ):
            shapeType = self.currentCollection[self.nextShapeLearnerToBeStarted]
            shapeType_code = self.nextShapeLearnerToBeStarted
            shape_index = self.indexOfShapeInCurrentCollection(shapeType)

            if usePrevParamsWhenShapeReappears \
               and self.shapeLearnersSeenBefore_currentCollection[self.nextShapeLearnerToBeStarted]:  #shape has been seen before
                [path, paramValues] = self.shapeLearners_currentCollection[shape_index].getLearnedShape()
                shapeLogger.info("%s: continuing learning. Current params: %s. Path: %s" % (shapeType, paramValues.flatten().tolist(), path.flatten().tolist()))
            else:
                [path, paramValues] = self.shapeLearners_currentCollection[shape_index].startFromScratch()
                #shapeLogger.info("%s: starting learning. Initial params: %s. Path: %s" % (shapeType, paramValues.flatten().tolist(), path.flatten().tolist()))

            paramsToVary = self.settings_shapeLearners_currentCollection[shape_index].paramsToVary
            self.nextShapeLearnerToBeStarted += 1
            shape = Shape(path=path, shapeID=0, shapeType=shapeType,
                          shapeType_code=shapeType_code, paramsToVary=paramsToVary, paramValues=paramValues)
            return shape
        else:
            raise RuntimeError('Don\'t know what shape learner you want me to start...')

    def feedbackManager(self, shapeIndex_messageFor, bestShape_index, noNewShape):
        shape_messageFor = self.shapeAtIndexInCurrentCollection(shapeIndex_messageFor)
        if (shape_messageFor < 0 ):
            shapeLogger.warning('Ignoring message because not for valid shape type')
            return -1
        else:

            if (noNewShape):  #just respond to feedback, don't make new shape
                self.shapeLearners_currentCollection[shapeIndex_messageFor].respondToFeedback(bestShape_index)
                return 1
            else:
                [numItersConverged, newPath, newParamValues] = self.shapeLearners_currentCollection[
                    shapeIndex_messageFor].generateNewShapeGivenFeedback(bestShape_index)
            paramsToVary = self.settings_shapeLearners_currentCollection[shapeIndex_messageFor].paramsToVary
            shape = Shape(path=newPath, shapeID=[], shapeType=shape_messageFor,
                          shapeType_code=shapeIndex_messageFor, paramsToVary=paramsToVary, paramValues=newParamValues)
            return numItersConverged, shape

    def respondToDemonstration(self, shapeIndex_messageFor, shape):
        shape_messageFor = self.shapeAtIndexInAllShapesLearnt(shapeIndex_messageFor)
        if (shape_messageFor < 0 ):
            shapeLogger.warning('Ignoring demonstration because not for valid shape type')
            return -1
        else:
            newPath, newParamValues, params_demo = self.shapeLearners_currentCollection[shapeIndex_messageFor].respondToDemonstration(shape)

            shapeLogger.info("%s: new demonstration.         Params: %s. Path: %s" % (shape_messageFor, params_demo.flatten().tolist(), shape.flatten().tolist()))
            logger_str = "%s: new demonstration.         Params: %s. Path: %s" % (shape_messageFor, params_demo.flatten().tolist(), shape.flatten().tolist())     
            self.currentDemo = logger_str
            paramsToVary = self.settings_shapeLearners_currentCollection[shapeIndex_messageFor].paramsToVary
            shape = Shape(path=newPath,
                          shapeID=[], 
                          shapeType=shape_messageFor,
                          shapeType_code=shapeIndex_messageFor, 
                          paramsToVary=paramsToVary, 
                          paramValues=newParamValues)
            shapeLogger.info("%s: new generated model.       Params: %s. Path: %s" % (shape_messageFor, newParamValues.flatten().tolist(), newPath.flatten().tolist()))
            logger_str = "%s: new generated model.       Params: %s. Path: %s" % (shape_messageFor, newParamValues.flatten().tolist(), newPath.flatten().tolist())           
            self.currentLearn = logger_str            
            return shape

    def respondToGoodDemonstration(self, shapeIndex_messageFor, shape):
        shape_messageFor = self.shapeAtIndexInAllShapesLearnt(shapeIndex_messageFor)
        if (shape_messageFor < 0 ):
            shapeLogger.warning('Ignoring demonstration because not for valid shape type')
            return -1
        else:
            response, newPath, newParamValues, params_demo = self.shapeLearners_currentCollection[shapeIndex_messageFor].respondToGoodDemonstration(shape)

            shapeLogger.info("%s: new demonstration.         Params: %s. Path: %s" % (shape_messageFor, params_demo.flatten().tolist(), shape.flatten().tolist()))

            paramsToVary = self.settings_shapeLearners_currentCollection[shapeIndex_messageFor].paramsToVary
            # shape.path would be an array of paths
            shape = Shape(path=newPath,
                          shapeID=[], 
                          shapeType=shape_messageFor,
                          shapeType_code=shapeIndex_messageFor, 
                          paramsToVary=paramsToVary, 
                          paramValues=newParamValues)
            shapeLogger.info("%s: new generated model.       Params: %s. Path: %s" % (shape_messageFor, newParamValues.flatten().tolist(), newPath.flatten().tolist()))
            return shape, response

    def respondToGoodDemonstration_modulo_phase(self, shapeIndex_messageFor, shape):
        shape_messageFor = self.shapeAtIndexInAllShapesLearnt(shapeIndex_messageFor)
        if (shape_messageFor < 0 ):
            shapeLogger.warning('Ignoring demonstration because not for valid shape type')
            return -1
        else:
            response, newPath, newParamValues, params_demo = self.shapeLearners_currentCollection[shapeIndex_messageFor].respondToGoodDemonstration_modulo_phase(shape)

            shapeLogger.info("%s: new demonstration.         Params: %s. Path: %s" % (shape_messageFor, params_demo.flatten().tolist(), shape.flatten().tolist()))

            paramsToVary = self.settings_shapeLearners_currentCollection[shapeIndex_messageFor].paramsToVary
            # shape.path would be an array of paths
            shape = Shape(path=newPath,
                          shapeID=[], 
                          shapeType=shape_messageFor,
                          shapeType_code=shapeIndex_messageFor, 
                          paramsToVary=paramsToVary, 
                          paramValues=newParamValues)
            shapeLogger.info("%s: new generated model.       Params: %s. Path: %s" % (shape_messageFor, newParamValues.flatten().tolist(), newPath.flatten().tolist()))
            return shape, response


    def indexOfShapeInCurrentCollection(self, shapeType):
        try:
            shapeType_index = self.currentCollection.index(shapeType)
        except ValueError:  #unknown shape
            shapeType_index = -1
        return shapeType_index

    def indexOfShapeInAllShapesLearnt(self, shapeType):
        try:
            shapeType_index = self.shapesLearnt.index(shapeType)
        except ValueError:  #unknown shape
            shapeType_index = -1
        return shapeType_index

    def shapeAtIndexInCurrentCollection(self, shapeType_index):
        try:
            shapeType = self.currentCollection[shapeType_index]
        except ValueError:  #unknown shape
            shapeType = -1
        return shapeType

    def shapeAtIndexInAllShapesLearnt(self, shapeType_index):
        try:
            shapeType = self.shapesLearnt[shapeType_index]
        except ValueError:  #unknown shape
            shapeType = -1
        return shapeType

    def shapesOfCurrentCollection(self):

        shapes = []

        for idx, shape_learner in enumerate(self.shapeLearners_currentCollection):

            path, paramValues = shape_learner.getLearnedShape()
            paramsToVary = shape_learner.paramsToVary
            shapeName = self.shapeAtIndexInCurrentCollection(idx)
            code = self.indexOfShapeInAllShapesLearnt(shapeName)

            shape = Shape(path=path,
                    shapeID=[], 
                    shapeType=shapeName,
                    shapeType_code=code, 
                    paramsToVary=paramsToVary, 
                    paramValues=paramValues)

            shapes.append(shape)

        return shapes

    def newCollection(self, collection):

        self.currentCollection = ""
        # check, for each letter, that we have the corresponding dataset
        for l in collection:
            print l

            self.generateSettings(l)
            '''
            try:
                self.generateSettings(l)
            except RuntimeError:
                raise RuntimeError( 'no dataset for this letter! <%s> '%l)
                shapeLogger.error("No dataset available for letter <%s>. Skipping this letter." % l)
                continue
            '''

            self.currentCollection += l

        self.nextShapeLearnerToBeStarted = 0

        shapeLogger.info("Starting to work on word <%s>" % collection)

        #print self.currentCollection

        try:
            self.collectionsLearnt.index(self.currentCollection)
            collectionSeenBefore = True
        except ValueError:
            collectionSeenBefore = False
            self.collectionsLearnt.append(self.currentCollection)

        self.initialiseShapeLearners()

        return collectionSeenBefore

    def resetParameterBounds(self, shapeType_index):
        currentBounds = self.shapeLearners_currentCollection[shapeType_index].getParameterBounds()

        #change bounds back to the initial ones 
        newBounds = self.shapeLearners_currentCollection[shapeType_index].initialBounds
        self.shapeLearners_currentCollection[shapeType_index].setParameterBounds(newBounds)
        shapeLogger.debug('Changing bounds on shape ' + self.shapeAtIndexInCurrentCollection(shapeType_index) + ' from ' + str(
            currentBounds) + ' to ' + str(newBounds))

    def generateSimulatedFeedback(self, shapeType_index, newShape, newParamValue):
        return self.shapeLearners_currentCollection[shapeType_index].generateSimulatedFeedback(newShape, newParamValue)

    def save_all(self, shapeIndex_messageFor):
        shape_messageFor = self.shapeAtIndexInAllShapesLearnt(shapeIndex_messageFor)
        if (shape_messageFor < 0):
            shapeLogger.warning('Ignoring demonstration because not for valid shape type')
            return -1
        else:
            self.shapeLearners_currentCollection[shapeIndex_messageFor].save_all()

    def save_demo(self, shapeIndex_messageFor):
        shape_messageFor = self.shapeAtIndexInAllShapesLearnt(shapeIndex_messageFor)
        if (shape_messageFor < 0):
            shapeLogger.warning('Ignoring demonstration because not for valid shape type')
            return -1
        else:
            self.shapeLearners_currentCollection[shapeIndex_messageFor].save_demo()

    def save_robot_try(self, shapeIndex_messageFor):
        shape_messageFor = self.shapeAtIndexInAllShapesLearnt(shapeIndex_messageFor)
        if (shape_messageFor < 0):
            shapeLogger.warning('Ignoring demonstration because not for valid shape type')
            return -1
        else:
            self.shapeLearners_currentCollection[shapeIndex_messageFor].save_robot_try()

    def save_params(self, shapeIndex_messageFor):
        shape_messageFor = self.shapeAtIndexInAllShapesLearnt(shapeIndex_messageFor)
        if (shape_messageFor < 0):
            shapeLogger.warning('Ignoring demonstration because not for valid shape type')
            return -1
        else:
            self.shapeLearners_currentCollection[shapeIndex_messageFor].save_params()
