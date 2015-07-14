"""
Class to use feedback to learn optimal parameters for a shape modeled by
an associated ShapeModeler.

Depends on shape_modeler and recordtype.
"""


import bisect
from copy import deepcopy

import numpy

from shape_modeler import ShapeModeler
from stroke import Stroke
import stroke

# shape learning parameters

# Allowed attempts to draw a new shape which is significantly different to the
# previous one but still within the range (just a precaution; sampling should
# be theoretically possible)
maxNumAttempts = 10000

# Tolerance on convergence test
tol = 1e-2

NUM_PRINCIPLE_COMPONENTS = 5

from recordtype import recordtype  #for mutable namedtuple (dict might also work)

SettingsStruct = recordtype('SettingsStruct',
                            ['shape_learning',  #String representing the shape which the object is learning
                             'initDatasetFile',  #Path to the dataset file that will be used to initialize the matrix for PCA
                             'updateDatasetFiles',  #List of path -- or single path-- to dataset that will be updated with demo shapes
                             'paramFile', #Path to the dataset file 'params.dat' inside which we save the learned parameters
                             'robotDataFiles', #List of path -- or single path-- to dataset that will be updated with robot tries
                             'paramsToVary',
                             #Natural number between 1 and number of parameters in the associated ShapeModeler, representing the parameter to learn
                             'doGroupwiseComparison',  #instead of pairwise comparison with most recent two shapes
                             'initialBounds',
                             #Initial acceptable parameter range (if a value is NaN, the initialBounds_stdDevMultiples setting will be used to set that value)
                             'initialBounds_stdDevMultiples',
                             #Initial acceptable parameter range in terms of the standard deviation of the parameter
                             'initialParamValue',
                             #Initial parameter value (NaN if to be drawn uniformly from initialBounds)
                             'minParamDiff']) #How different two shapes' parameters need to be to be published for comparison
#@todo: make groupwise comparison/pairwise comparison different implementations of shapeLearner class

class ShapeLearner:
    def __init__(self, settings):

        self.paramsToVary = settings.paramsToVary
        #self.numPrincipleComponents = max(self.paramsToVary)
        self.numPrincipleComponents = NUM_PRINCIPLE_COMPONENTS
        #assign a ShapeModeler to use
        print(settings.initDatasetFile)
        self.shapeModeler = ShapeModeler(init_filename=settings.initDatasetFile,
                                         update_filenames=settings.updateDatasetFiles,
                                         param_filename=settings.paramFile,
                                         robot_filenames=settings.robotDataFiles,
                                         num_principle_components=self.numPrincipleComponents)

        self.bounds = settings.initialBounds
        for i in range(len(self.paramsToVary)):
            parameterVariances = self.shapeModeler.getParameterVariances()
            if (numpy.isnan(settings.initialBounds[i, 0]) or numpy.isnan(
                    settings.initialBounds[i, 1])):  #want to set initial bounds as std. dev. multiple
                boundsFromStdDevMultiples = numpy.array(settings.initialBounds_stdDevMultiples[i, :]) * \
                                            parameterVariances[settings.paramsToVary[i] - 1]

                if (numpy.isnan(settings.initialBounds[i, 0])):
                    self.bounds[i, 0] = boundsFromStdDevMultiples[0]
                if (numpy.isnan(settings.initialBounds[i, 1])):
                    self.bounds[i, 1] = boundsFromStdDevMultiples[1]

        self.doGroupwiseComparison = settings.doGroupwiseComparison
        self.shape_learning = settings.shape_learning
        self.minParamDiff = settings.minParamDiff

        self.initialParamValue = settings.initialParamValue
        self.params = numpy.zeros((self.numPrincipleComponents, 1))
        self.shape = self.shapeModeler.makeShape(self.params) # <--- now this is what matters (because of multi-stroke)

        for i in range(self.numPrincipleComponents):
            self.params[i][0] = -self.initialParamValue[i]

        self.initialBounds = deepcopy(self.bounds)
        self.converged = False
        self.numIters = 0
        self.numItersConverged = 0

    ### ----------------------------------------------------- START LEARNING
    def startLearning(self):
        #make initial shape
        if (numpy.isnan(self.initialParamValue[0])):
            [shape, paramValues] = self.shapeModeler.makeRandomShapeFromUniform(self.params, self.paramsToVary,
                                                                                self.bounds)
            self.params = paramValues
        else:
            shape = self.shapeModeler.makeShape(self.params)

        self.bestParamValue = self.params[
            self.paramsToVary[0] - 1]  #   USE ONLY FIRST PARAM IN LIST FOR SELF-LEARNING ALGORITHM

        if (self.doGroupwiseComparison):
            self.params_sorted = [self.bounds[0, 0],
                                  self.bounds[0, 1]] #   USE ONLY FIRST PARAM IN LIST FOR SELF-LEARNING ALGORITHM
            bisect.insort(self.params_sorted, self.bestParamValue)
            self.shapeToParamsMapping = [self.params]
        else:
            self.newParamValue = self.bestParamValue
            self.params = [self.newParamValue]

        return shape, self.params

    ### ---------------------------------------- START FROM BASIC LINE OR LAST LEARNED SHAPE
    def startFromScratch(self):
        shape = self.shapeModeler.readStartingPoint()
        self.params,_ = self.shapeModeler.decomposeShape(shape)
        return shape, self.params

    ### ---------------------------------------- START LEARNING - TRIANGULAR
    def startLearningAt(self, startingBounds, startingParamValues):
        self.bounds = startingBounds

        #make initial shape
        [shape, paramValues] = self.shapeModeler.makeRandomShapeFromriangular(self.params, self.paramsToVary,
                                                                              self.bounds, startingParamValues)
        self.params = paramValues
        self.bestParamValue = paramValues[
            self.paramsToVary[0] - 1]  #   USE ONLY FIRST PARAM IN LIST FOR SELF-LEARNING ALGORITHM
        if (self.doGroupwiseComparison):
            self.params_sorted = [self.bounds[0, 0],
                                  self.bounds[0, 1]] #   USE ONLY FIRST PARAM IN LIST FOR SELF-LEARNING ALGORITHM
            bisect.insort(self.params_sorted, self.bestParamValue)
            self.shapeToParamsMapping = [self.params]
        else:
            self.newParamValue = self.bestParamValue

        return shape, self.bestParamValue


    ### ----------------------------------------------- MAKE DIFFERENT SHAPE
    def makeShapeDifferentTo(self, paramValue):
        #make new shape to compare with
        [newShape, newParams] = self.shapeModeler.makeRandomShapeFromTriangular(self.params, self.paramsToVary,
                                                                                self.bounds, [
                paramValue])  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
        newParamValue = newParams[self.paramsToVary[0] - 1, 0] #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
        #ensure it is significantly different
        numAttempts = 1
        while (abs(newParamValue - paramValue) < self.minParamDiff and numAttempts < maxNumAttempts):
            [newShape, newParams] = self.shapeModeler.makeRandomShapeFromTriangular(self.params, self.paramsToVary,
                                                                                    self.bounds, [
                    paramValue]) #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
            newParamValue = newParams[
                self.paramsToVary[0] - 1, 0] #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
            numAttempts += 1

        if (numAttempts >= maxNumAttempts):  #couldn't find a 'different' shape in range
            print('Oh no!')  #this should be prevented by the convergence test below

        #store it as an attempt
        if (self.doGroupwiseComparison):
            bisect.insort(self.params_sorted, newParamValue)
            self.shapeToParamsMapping.append(newParams)

        return newShape, newParamValue

    ### ------------------------------------------------- MAKE SIMILAR SHAPE
    def makeShapeSimilarTo(self, paramValue):
        #make new shape, but don't enforce that it is sufficiently different
        [newShape, newParamValues] = self.shapeModeler.makeRandomShapeFromTriangular(self.params, self.paramsToVary,
                                                                                     self.bounds, [
                paramValue])  #USE FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
        newParamValue = newParamValues[self.paramsToVary[0] - 1, 0] #USE FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM

        #store it as an attempt
        if (self.doGroupwiseComparison):
            bisect.insort(self.params_sorted, newParamValue)
            self.shapeToParamsMapping.append(newParamValues)

        return newShape, newParamValue

    ### ---------------------------------------- GENERATE SIMULATED FEEDBACK
    def generateSimulatedFeedback(self, shape, newParamValue):
        #code in place of feedback from user: go towards goal parameter value
        goalParamValue = numpy.float64(0) #-1.5*parameterVariances[self.paramToVary-1]
        goalParamsValue = numpy.zeros((self.numPrincipleComponents, 1))
        goalParamsValue[self.paramToVary - 1, 0] = goalParamValue
        if (self.doGroupwiseComparison):
            errors = numpy.ndarray.tolist(abs(self.shapeToParamsMapping - goalParamsValue))
            bestShape_idx = errors.index(min(errors))
        else:
            errors = [abs(self.bestParamValue - goalParamValue), abs(newParamValue - goalParamValue)]
            bestShape_idx = errors.index(min(errors))
        return bestShape_idx

    ### ------------------------------------------------ RESPOND TO FEEDBACK
    def respondToFeedback(self, bestShape):
        #update bestParamValue based on feedback received
        if (self.doGroupwiseComparison):
            params_best = self.shapeToParamsMapping[bestShape]

            self.bestParamValue = params_best[
                self.paramsToVary[0] - 1, 0]  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
            bestParamValue_index = bisect.bisect(self.params_sorted,
                                                 self.bestParamValue) - 1  #indexing seems to start at 1 with bisect
            newBounds = [self.params_sorted[bestParamValue_index - 1], self.params_sorted[bestParamValue_index + 1]]

            #restrict bounds if they were caused by other shapes, because it must be sufficiently different to said shape(s)
            if ((bestParamValue_index - 1) > 0):  #not the default min
                newBounds[0] += self.minParamDiff
            if ((bestParamValue_index + 1) < (len(self.params_sorted) - 1)):  #not the default max
                newBounds[1] -= self.minParamDiff

            if (not (newBounds[0] > newBounds[1])):  #protect from bounds switching expected order
                self.bounds[0, :] = newBounds #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM

            diff_params = params_best - self.params
            diff = numpy.linalg.norm(diff_params)
            self.params += diff_params / 2
        else:  #do pairwise comparison with most recent shape and previous
            #restrict limits
            if ( bestShape == 'new' ):  #new shape is better
                worstParamValue = self.bestParamValue
                bestParamValue = self.newParamValue
                self.params[self.paramToVary - 1, 0] = bestParamValue
            else:  #new shape is worse
                worstParamValue = self.newParamValue
                bestParamValue = self.bestParamValue
                self.params[self.paramToVary - 1, 0] = bestParamValue

            if ( worstParamValue == min(self.bestParamValue, self.newParamValue) ):  #shape with lower value is worse
                self.bounds[0] = worstParamValue  #increase min bound to worst so we don't try any lower
            else:  #shape with higher value is worse
                self.bounds[1] = worstParamValue  #decrease max bound to worst so we don't try any higher

            ### ------------------------------------------------------------ ITERATE

    def generateNewShapeGivenFeedback(self, bestShape):
        #------------------------------------------- respond to feedback
        self.respondToFeedback(bestShape)  #update bounds and bestParamValue

        #----------------------------------------- check for convergence
        #continue if there are more shapes to try which are different enough
        if ((abs(self.bounds[0, 1] - self.bestParamValue) - self.minParamDiff < tol) and (abs(
                    self.bestParamValue - self.bounds[
                    0, 0]) - self.minParamDiff) < tol):  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
            self.converged = True
        else:
            self.converged = False


        #-------------------------------------------- continue iterating
        self.numIters += 1

        #try again if shape is not good enough
        if (not self.converged):
            self.numItersConverged = 0
            [newShape, newParamValue] = self.makeShapeDifferentTo(self.bestParamValue)
            self.newParamValue = newParamValue
            return self.numItersConverged, newShape, newParamValue

        else:
            self.numItersConverged += 1
            [newShape, newParamValue] = self.makeShapeSimilarTo(self.bestParamValue)
            self.newParamValue = newParamValue
            return self.numItersConverged, newShape, newParamValue

    def getLearnedParams(self):
        return self.params

    def getLearnedShape(self):
        #return self.shapeModeler.makeShape(self.params), self.params
        return self.shape, self.params

    def getParameterBounds(self):
        return self.bounds

    def setParameterBounds(self, bounds):
        self.bounds = bounds

    def respondToDemonstration(self, shape):
        """
        Algo:
        
        1) takes the shape of the demonstration
        
        2) takes the curent learned shape
        
        3) re-performs PCA taking in account the domonstrated shape,
           then obtains a new space with new eigen vectors
           
        4) projects demonstrated and learned shapes into this new space 
           and gets their new parameters  
           
        5) updates the learned parameters as the algebric middle 
           between demonstrated parameters and curent learned parameters. 
        """
        demo_shape = ShapeModeler.normaliseShapeWidth(numpy.array(shape))
        
        # take the shape corresponding to the curent learned parameters in the curent space
        learned_shape = self.shapeModeler.makeShape(self.params)
        
        # add the demo shape to the matrix for PCA and re-compute reference-shape params
        self.shapeModeler.extendDataMat(demo_shape)
        ref_params = self.shapeModeler.refParams
        
        # re-compute parameters of the learned shape and the demo shape in the new PCA-space
        params_demo, _ = self.shapeModeler.decomposeShape(demo_shape)
        self.params, _ = self.shapeModeler.decomposeShape(learned_shape)
        #d to get distance with clusters
        
        # learning :
        diff_params = params_demo - self.params
        self.params += 0*diff_params/2 #go towards the demonstrated shape
        #self.params = params_demo

        self.shape = 0.5*(demo_shape+self.shape)


        #self.params[self.paramsToVary[0]-1] = params_demo[self.paramsToVary[0]-1] #ONLY USE FIRST PARAM
        #store it as an attempt (this isn't super appropriate but whatever)
        """if (self.doGroupwiseComparison):
            newParamValue = self.params[
                self.paramsToVary[0] - 1, 0]  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
            #print('Demo params: '+str(self.params))
            bisect.insort(self.params_sorted, newParamValue)
            self.shapeToParamsMapping.append(self.params)
            #self.respondToFeedback(len(self.params_sorted)-3) # give feedback of most recent shape so bounds modify"""
        return demo_shape, self.params, params_demo
        #return self.shapeModeler.makeShape(self.params), self.params, params_demo

    def respondToGoodDemonstration(self, shape):
        """
        will learn the demo only if it is close to the goal shape, the way it was drawn is important
        return a number 0, 1 or 2:
        0 = the demo shape has no meaning, not accepted
        1 = the demo shape is close to the ref, but was not drawn in the good way
        2 = the demo shape is close to the fer, accepted
        """
        """
        Algo:
        
        1) takes the shape of the demonstration and check if it is good enought to be learned
        
        2) takes the curent learned shape
        
        3) re-performs PCA taking in account the domonstrated shape,
           then obtains a new space with new eigen vectors
           
        4) projects demonstrated and learned shapes into this new space 
           and gets their new parameters  
           
        5) updates the learned parameters as the algebric middle 
           between demonstrated parameters and curent learned parameters. 
        """
        demo_shape = ShapeModeler.normaliseShapeWidth(numpy.array(shape))
        reference = self.shapeModeler.getReference()

        # create stroke from shape :
        demo_stroke = Stroke()
        demo_stroke.stroke_from_xxyy(numpy.reshape(demo_shape,len(demo_shape)))
        demo_stroke.uniformize()
        # create stroke from reference :
        ref_stroke = Stroke()
        ref_stroke.stroke_from_xxyy(reference)
        ref_stroke.uniformize()
        # get distance between demo and ref :
        _,dist1 = stroke.euclidian_distance(demo_stroke,ref_stroke)
        # get distance between demo and ref, modulo way and phase :
        new_x,new_y,_,dist2,_,_ = stroke.best_aligment(demo_stroke,ref_stroke)
        #demo_shape = numpy.concatenate((new_x,new_y),axis=0)
        #demo_shape = numpy.reshape(demo_shape, (-1, 1))

        response = 2

        if dist2>0.3:
            response = 0

        if dist1>0.3 and dist2<=0.3:
            response = 1

        # take the shape corresponding to the curent learned parameters in the curent space
        learned_shape = self.shapeModeler.makeShape(self.params)
        
        # if good shape, add the demo shape to the matrix for PCA and re-compute reference-shape params
        if response==2:
            self.shapeModeler.extendDataMat(demo_shape)
        ref_params = self.shapeModeler.refParams
        
        # re-compute parameters of the learned shape and the demo shape in the new PCA-space
        params_demo, _ = self.shapeModeler.decomposeShape(demo_shape)
        print 'parameters of demo : '
        print params_demo
        self.params, _ = self.shapeModeler.decomposeShape(learned_shape)
        #d to get distance with clusters
        
        # learning, if good shape :
        diff_params = params_demo - self.params
        if response==2:
            self.shape = 0.5*(self.shape+demo_shape)
            self.params += diff_params/2 #go towards the demonstrated shape

        #self.params[self.paramsToVary[0]-1] = params_demo[self.paramsToVary[0]-1] #ONLY USE FIRST PARAM
        #store it as an attempt (this isn't super appropriate but whatever)
        """if (self.doGroupwiseComparison):
            newParamValue = self.params[
                self.paramsToVary[0] - 1, 0]  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
            #print('Demo params: '+str(self.params))
            bisect.insort(self.params_sorted, newParamValue)
            self.shapeToParamsMapping.append(self.params)
            #self.respondToFeedback(len(self.params_sorted)-3) # give feedback of most recent shape so bounds modify"""
        return response, self.shapeModeler.makeShape(self.params), self.params, params_demo
        #return response, self.shape, self.params, params_demo
    
    def respondToGoodDemonstration_modulo_phase(self, shape):
        """
        will learn the demo only if it is close to the goal shape, but no matter the way it was drawn
        return a boolean that says if the stroke was accepted or not
        """
        """
        Algo:
        
        1) takes the shape of the demonstration and check if it is good enought to be learned
        
        2) takes the curent learned shape
        
        3) re-performs PCA taking in account the domonstrated shape,
           then obtains a new space with new eigen vectors
           
        4) projects demonstrated and learned shapes into this new space 
           and gets their new parameters  
           
        5) updates the learned parameters as the algebric middle 
           between demonstrated parameters and curent learned parameters. 
        """
        demo_shape = ShapeModeler.normaliseShapeWidth(numpy.array(shape))
        reference = self.shapeModeler.getReference()

        # create stroke from shape :
        demo_stroke = Stroke()
        demo_stroke.stroke_from_xxyy(numpy.reshape(demo_shape,len(demo_shape)))
        demo_stroke.uniformize()
        # create stroke from reference :
        ref_stroke = Stroke()
        ref_stroke.stroke_from_xxyy(reference)
        ref_stroke.uniformize()
        # get distance between demo and ref, modulo way and phase :
        new_x,new_y,_,dist,_,_ = stroke.best_aligment(demo_stroke,ref_stroke)
        demo_shape = numpy.concatenate((new_x,new_y),axis=0)
        demo_shape = numpy.reshape(demo_shape, (-1, 1))

        accepted = True

        if dist>0.4:
            accepted = False

        # take the shape corresponding to the curent learned parameters in the curent space
        learned_shape = self.shapeModeler.makeShape(self.params)
        
        # if good shape, add the demo shape to the matrix for PCA and re-compute reference-shape params
        if accepted:
            self.shapeModeler.extendDataMat(demo_shape)
        ref_params = self.shapeModeler.refParams
        
        # re-compute parameters of the learned shape and the demo shape in the new PCA-space
        params_demo, _ = self.shapeModeler.decomposeShape(demo_shape)
        self.params, _ = self.shapeModeler.decomposeShape(learned_shape)
        #d to get distance with clusters
        
        # learning, if good shape :
        diff_params = params_demo - self.params
        if accepted:
            #self.params = params_demo
            self.params += diff_params/2 #go towards the demonstrated shape

            # check new generated shape :
            new_shape = self.shapeModeler.makeShape(self.params)
            new_stroke = Stroke()
            new_stroke.stroke_from_xxyy(numpy.reshape(demo_shape,len(new_shape)))
            _,dist1 = stroke.euclidian_distance(new_stroke,ref_stroke)

            if dist1>0.4:
                self.params = params_demo

        #self.params[self.paramsToVary[0]-1] = params_demo[self.paramsToVary[0]-1] #ONLY USE FIRST PARAM
        #store it as an attempt (this isn't super appropriate but whatever)
        '''if (self.doGroupwiseComparison):
            newParamValue = self.params[
                self.paramsToVary[0] - 1, 0]  #USE ONLY FIRST PARAM FOR SELF-LEARNING ALGORITHM ATM
            #print('Demo params: '+str(self.params))
            bisect.insort(self.params_sorted, newParamValue)
            self.shapeToParamsMapping.append(self.params)
            #self.respondToFeedback(len(self.params_sorted)-3) # give feedback of most recent shape so bounds modify'''
        return accepted, self.shapeModeler.makeShape(self.params), self.params, params_demo

    def save_all(self):
        self.shapeModeler.save_all()
    
    def save_demo(self):
        self.shapeModeler.save_demo()

    def save_params(self):
        paramValue = []
        for i in range(self.numPrincipleComponents):
             paramValue.append(-self.params[i][0])
        self.shapeModeler.save_params(paramValue, self.shape_learning)

    def save_robot_try(self):
        self.shapeModeler.save_robot_try(self.shapeModeler.makeShape(self.params).T)
