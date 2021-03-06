"""
    Driver codes for Capstone Group 27 project experimentation.
    There are drivers to run and evaluate the models when subject
    to different adversarial attacks and defenses. There are also
    drivers to generate images with adversarial perturbations for
    display purposes.
"""



# imports
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from driver_utilities import singleImageTransformForBatch,formatTimeDiff
from io_methods import plot_acc,write_graph_data



# constants:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_PERIOD = 50



# methods
def runTest(attack:Any,model:torch.nn.Module,testLoader:DataLoader,eRange:List[float]=None,defense:Any=None,passE:bool=True,constE:float=0.0,\
            numBatchesToRun:int=None,singleImage_atk:bool=False,singleImage_def:bool=False,writeToFile:bool=False,filePath:str=None,\
            graphLabel:str=None,graphColor:str=None,seqNum:int=None,atkargs:Dict=None,defargs:Dict=None):
    r"""
        Runs attacks with the given range and then plots the obtained accuracy and writes the data into txt file. This is the primary driver
        for our experimentations. This method will run attacks for each epoch and record the obtained accuracy in a list. During this process time
        will also be kept track of. The accuracy and time for each epoch will be printed. After all of the epochs are run, recorded data will be plotted
        and written into a text file.
        -----------------------------
        @params:
            - attack: Attack function, can be imported from attacks.py
            - model: Pre-trained model upon which to run the attacks on
            - testLoader: Dataset encapsulated in torch.utils.data.DataLoader object with pre-processing transformation already applied
            - eRange: List of desired Normalized L2 Dissimilarity factors to run the attacks with. Each dissimilarity factor correlates to a single epoch.
            - defense: Transformation for defense, can be imported from defenses.py.
            - passE: If True then a list of L2 Dissimilarity Values will be passed through eRange paramter otherwise a single constant dissimilarity factor will be
                            used. If False, then only a single epoch will be run using the constant dissimilarity factor.
            - constE: constant dissimilarity factor passed in if passE is False.
            - numBatchesToRun: Specify how many batches should be run. By default all batches are run, however, for testing purposes less can be run.
            - singleImage_atk: If True, then each image has to be transformed one at a time. Transforming one at a time requires additional processing and time.
            - singleImage_def: If True, then each image has to be transformed one at a time. Transforming one at a time requires additional processing and time.
            - writeToFile: If True, then the obtained data will be written to a text file.
            - filePath: Path to the file where results will be written.
            - graphLabel: Label for the graph, specifically for the line generated by the current results. The name should correlate to the defensive transformation.
            - graphColor: Either name of common colors or the hex color representation. The color will correlate to the line graph generated from the data obtained here.
            - seqNum: Used for splitting up parts of the experimentation. Subsets of the overall dissimilarity factors are run at different times and places and seqNum keeps track
                            of the range being used. seqNum correlates to increases set of ranges, if the range was [1,2,3,4] then [1,2] would have seqNum 1 and [3,4] would have.
                            seqNum 2. This parameter does not have to be used if running all epoch in one place.
            - atkargs: A dictionary of additional keyword arguments for attack transformations. See attack transformations in attacks.py to see what can/should be passed in
                            through this parameter.
            - defargs: Same as atkargs but for defenses.
    """
  
    # set default values
    if isinstance(eRange,int):
        eRange_inner = range(eRange)
    else:
        eRange_inner = np.arange(0,0.08,0.005) if eRange is None else eRange
    filePath = 'changePathLater_graphData.txt' if filePath is None else filePath
    graphLabel = 'unlabeled' if graphLabel is None else graphLabel
    graphColor = 'gray' if graphColor is None else graphColor

    accuracies = []
    timeAccumulator = 0
    for e in eRange_inner:
        # start timer:
        prevTime = time.time()
        if passE:
            newAcc = runAttack(attack,model,testLoader,e,defense=defense,\
                        numBatchesToRun=numBatchesToRun,singleImage_atk=singleImage_atk,\
                        singleImage_def=singleImage_def,atkargs=atkargs,defargs=defargs)
        else:
            newAcc = runAttack(attack,model,testLoader,constE,defense=defense,\
                        numBatchesToRun=numBatchesToRun,singleImage_atk=singleImage_atk,\
                        singleImage_def=singleImage_def,atkargs=atkargs,defargs=defargs)
    
        newTime = time.time()
        timeDiff = newTime - prevTime
        timeAccumulator += timeDiff
        formattedTimeDiff = formatTimeDiff(timeDiff)
    
        accuracies.append(newAcc)
        print(f"epoch {e} complete with acc {newAcc} in {formattedTimeDiff}")
  
    plot_acc(accuracies,eRange)
    if writeToFile:
        write_graph_data(accuracies,graphLabel,graphColor,filePath,x_range=eRange,seqNum=seqNum)
    timeAccumulator = formatTimeDiff(timeAccumulator)
    print(f"Total elapsed time: {timeAccumulator}")

def runAttack(attack:Any,model:torch.nn.Module,testLoader:DataLoader,e:float=0.0,defense:Any=None,numBatchesToRun:int=None,\
              singleImage_atk:bool=False,singleImage_def:bool=False,atkargs:Dict=None,defargs:Dict=None)->float:
    r"""
        Runs the provided attack with the provided resources. Runs attack on single
        batch. Note, it is not necessary to pass in anything for atkargs or defargs.

        --------------------
        @param:
            - attack: the attack to run 
            - model: the model to run the attack on
            - testLoader: the data that will be used in the attack
            - defense: method that runs the defensive transformation
            - numBatchesToRun: the number of batches to run before qutting the process; used for testing
            - singleImage_atk/def: whether or not the transformation is done one image at a time; call to singleImageTransformForBatch() depends on this 
            - atkargs: extra parameters that are specific to the attack; look at
                        named parameters in the attack and pass them in through this 
                        method. NOTE: This needs to be passed in as a Dictionary Object.
            - defargs: a dictionary of arguments for defensive transformations; dictionary keys and values should match what is expected from
                        defensive technique being used. NOTE: This needs to be passed in as a Dictionary Object.
    
        ------------------
        @returns:
            - accuracy: of the attack
    """

    # model adjustments:
    model.to(device)
    model.eval()

    # def args initialization (if necessary):
    if defargs is None:
        defargs = {}
    if atkargs is None:
        atkargs = {}

    #def accumulators:
    numImages = 0
    numCorrects = 0
    count = 0



    for img,target in testLoader:
        # move images and labels to gpu
        img,target = img.to(device),target.to(device)
    
    
        # apply attack transformation if one is given
        if attack is not None:# generate attack images using attack algos; different cases based on wheter there are any key work args available
      
            model.eval() # set model to evaluate at the start of the attack
      
            if singleImage_atk: # primarily for deepfool
                atkargs['e'] = e
                atkImg = singleImageTransformForBatch(img,attack,reqModel=True,model=model,targets=target,**atkargs)
            else:
                atkImg = attack(model,img,target,e=e,**atkargs)
        else: #in the cases where we are not using any attacks; this edit was made later, which is why the term atkImg is still used
            atkImg = img
    
        #check req for results that return attack image and perturbation:
        if isinstance(atkImg,tuple):
            atkImg = atkImg[0]
    
        #after attack is over zero out model's grad and set model to train:
        model.zero_grad()
        model.train()
        atkImg = atkImg.clone().detach()
        atkImg.requires_grad = False

        # apply defensive transformation
        # some attack algo return 3-dim images on those cases a 4th will be added to be able to work with models
        if (len(atkImg.size()) == 3):
            atkImg = atkImg.unsqueeze(0)   
        if defense is not None:
            if singleImage_def:
                atkImg = singleImageTransformForBatch(atkImg,defense,**defargs)
            else:
                atkImg = defense(atkImg,**defargs)
    
        # get and process the scores for the attack images  
        atkScores = model(atkImg)
        atkPred = torch.max(atkScores,1)[1]

        #update accumulator:
        numImages += len(img)
        numCorrects += torch.sum(atkPred==target).item()

        #check if loop should be broken:
        count += 1
        if count % BATCH_PERIOD == 0:
            print(f"Batch {count} completed. Batch accuracy: {torch.sum(atkPred==target).item()/len(img)}. Accumulated accuracy: {numCorrects/numImages}")
        if numBatchesToRun is not None:
            if count == numBatchesToRun:
                break
  
    accuracy = numCorrects/numImages
    return accuracy
