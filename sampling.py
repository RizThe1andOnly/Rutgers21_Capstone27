"""
    Methods to sample pictures with one of various adversarial transformations applied to
    them.
"""



from typing import Any, List, Tuple, Union
import torch
from torch.tensor import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np



#constants:
device = 'cuda:0'



# methods:
def showSampleImages(samples:List[Tuple[Tensor,Tensor]],numImagesGiven:int=None):
    r"""
        Displays images and their adversarial counterparts provided by another method that generates
        the adversarial images.

        samples is a list of tuples with the following elements:
            - [0]: the clean image stuff
                - [0]: the image tensor
                - [1]: the image label
            - [1]: the adversarial image stuff; same structure as [0]
    """
    numImages = len(samples)
    numImagesGiven = numImagesGiven if numImagesGiven is not None else len(samples)

    fig,axes = plt.subplots(numImages,2)
    fig.set_figheight(1*numImages)
    fig.set_figwidth(5)
    fig.tight_layout(pad=1)

    for i in range(min(numImages,numImagesGiven)):
        currentCleanImageAxis = axes[i][0]
        currentAdvImageAxis = axes[i][1]

        cleanSample = samples[i][0]
        advSample = samples[i][1]

        cleanImage = transforms.functional.to_pil_image(cleanSample[0].squeeze())
        advImage = transforms.functional.to_pil_image(advSample[0].squeeze())


        cleanLabel = cleanSample[1]
        advLabel = advSample[1]

        cleanImageTitle = "True Label : " + str(cleanLabel)
        advImageTitle = "Adversarial Label : " + str(advLabel)

        currentCleanImageAxis.set_title(cleanImageTitle)
        currentCleanImageAxis.imshow(cleanImage,interpolation = 'none')

        currentAdvImageAxis.set_title(advImageTitle)
        currentAdvImageAxis.imshow(advImage,interpolation='none')

def collectDifferentDissimilarSamples(attack:Any,model:Module,testingData:DataLoader,dissimRange:Union[np.ndarray,List]=None,samplesToBeCollected:int=10,**kwargs):
    r"""
        Runs attack on the images for each of the values inside dissimRange. From there
        images will be collected and displayed. The goal here is to collect images and 
        their corresponding perturbations for presentation purposes.

        !!!Note!!! : The batchsize for the Dataloader has to be 1 otherwise errors will occur.

        Note: Since there are many cases where the perturbations are very prevelant we
        will collect samples where the perturbations are not very detectable (to humans)
        and present those.

        Note: See [1] in README.md -> References for details on Dissimilarity.

        ----------------------
        @params:
            - attack: The function for the attack to run.
            - model: The model which the attack is to be run against.
            - testingData:  The Dataloader object with the images and targets that are to be run with the attack.
                            The batch size should be set to 1, otherwise this method will FAIL.
            - dissimRange:  The range which includes the dissimilarity values; these determine the strength of the perturbations.
                            The default will be set to (None) which will be np.arange(0.01,0.1,0.02). If passing in custom range
                            then use the np.arange() method. Note that the range starts at 0.01 beacuse disSimilarity = 0.0 is
                            the original image and thus will not be necessary to do calculations upon.
            - **kwargs: The parameters to be put into the attack function; see the attack function
                        to see what should go here.
    """

    # set items to gpu:
    model.to(device)
  
    # set the default dissimRange:
    dissimRange = np.arange(0.02,0.1,0.02) if dissimRange is None else dissimRange

    # run attack iteration for each value in dissimRange:
    sampleImageCount = 0
    sampleList = []
    for img,target in testingData:
        img,target = img.to(device),target.to(device)
        origSet = (img,torch.zeros(img.size(),device=device),0.0,target.item())
        attackSetList = []
        incorrectStreak = 0
        for e in dissimRange:

            # run the images through attack and obtain the attack image
            if len(kwargs.items()) > 0:
                atkImg,perturbation = attack(model,img,target,e,**kwargs)
            else:
                atkImg,perturbation = attack(model,img,target,e)
      
      
            if len(atkImg.size()) == 3:
                atkImg = torch.unsqueeze(atkImg,0)
      
            # get the prediction of the attack image
            atkScore = model(atkImg)
            atkPred = torch.max(atkScore,1)[1].item()
      
            # add the attack image and the perturbation to a candidate list containing the image at diff dissim factors
            attackSet = (atkImg,perturbation,e,atkPred)
            attackSetList.append(attackSet)

            # see if the attack image's prediction matches the target or not; an atk image has to miss the target at every step to be returned.
            if atkPred != target.item():
                incorrectStreak += 1
      
        # check if image and perturbations qualify to be added to the sampleList:
        if incorrectStreak == len(dissimRange):
            sample = [origSet]
            for x in attackSetList:
                sample.append(x)
            sampleList.append(sample)
            sampleImageCount += 1
            if sampleImageCount == samplesToBeCollected:
                break
    
    # present the images using helper method:
    _present_image_range(sampleList)

def _present_image_range(sampleList:List[List[Tensor]]):
    r"""
        Uses Matplotlib to show the samples collected through 
        collectDifferentDissimilarSamples(). This is a private method to
        be used only with collectDifferentDissimilarSamples(). The sampleList
        parameter here gets the argument of the same name from collectDifferentDissimilarSamples().
        See that variable and that function to see what this parameter is like and how to 
        process it.
        ---------------------
        param:
            - sampleList: List of lists. Each inner list contains images with dissimilarity factors ranging from 0 to 0.075.
                        See [1] in README.md->References to see details on dissimilarity.
    """

    # get dimensions:
    numImages = len(sampleList)
    numRange = len(sampleList[0][1]) + 1 # plus 1 for the original image itself

    # create the graphs that images will be displayed in:
    fig,axes = plt.subplots(numImages*2,numRange)
    fig.set_figheight(2*numImages)
    fig.set_figwidth(5)
    fig.tight_layout(pad=1)

    #display the images:
    for i in range(numImages):
        for j in range(numRange):
            # set the current axis and data to be displayed
            sampleIndex = i * 2 
            currentAxis = axes[sampleIndex][j]
            currentAxis.get_xaxis().set_visible(False)
            currentPerturbationAxis = axes[sampleIndex+1][j]
            currentPerturbationAxis.get_xaxis().set_visible(False)
            toBeDisplayed = sampleList[i][j]
      
            # extract the specific data to be displayed
            image = toBeDisplayed[0]
            perturbation = toBeDisplayed[1]
            e_val = toBeDisplayed[2]
            classificationLabel = toBeDisplayed[3]

            #obtain perturbations for dissim value > 0:
            if j > 0:
                perturbation[perturbation > torch.mean(perturbation)] = perturbation[perturbation > torch.mean(perturbation)] * 10


            #construct the image from the tensor:
            image = transforms.functional.to_pil_image(image[0].cpu().squeeze())
            perturbation = transforms.functional.to_pil_image(perturbation[0].cpu().squeeze())

            title = str(e_val) + ' ('+str(classificationLabel)+')'
            currentAxis.set_title(title)
            currentAxis.imshow(image,interpolation='none')
            currentPerturbationAxis.imshow(perturbation,interpolation='none')

