"""
    Methods that are used for (re-)training a model obtained from torchvision.models.
"""



# imports
from typing import Any
from os import path
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from driver_utilities import formatTimeDiff,singleImageTransformForBatch
import time



#constants:
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_PARAMETERS = 'model_parameters'
OPTIMIZER_PARAMETERS = 'optimizer_parameters'
EPOCH_COUNT = 'epoch_count'
LOSS = 'loss'



# methods:
def trainer(model:Module, trainloader:DataLoader, epochs:int, transform:Any,testLoader:DataLoader,filePath:str=None)->Module:
    r"""
        Trains a model with a given transform (defensive). trainLoader should be a 
        dataloader that should use the "training" dataset; when loading the dataset
        the "train" parameter should be set to True.
        ----------------------
        params:
            - model: model to be trained; a pre-trained model loaded from Pytorch's Torchvision
            - trainLoader: Dataloader with training data; the train variable in dataset method was set to True
            - epochs: number of iterations to train the model for
            - transform: transform (defensive) with which to trian the model with
            - testLoader: Dataloader with test data, used for validation purposes. Validation will only be run if
                            something is passed in for this parameter.
            - filePath: Path for file in which to save the re-trained model's parameters.
        ----------------------
        returns:
            - model: the re-trained model
    """
    model = model.to(device)
    ce_loss = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = optim.Adam(params=params,lr=.0001)

    for e in range(epochs):
        if not e % 10: print(e)
        for imgs, lbls in trainloader:
            image = imgs.to(device)
            label = lbls.to(device)
          
            trans_img = transform(image)

            output = model(trans_img)
            model.zero_grad()
            loss = ce_loss(output,label)
            loss.backward()
          
            optimizer.step()
          
  
    filePath = 'some_model.pth' if filePath is None else filePath
    torch.save(model.state_dict(), filePath)
    #files.download(filePath) # used with google colab
    
    if testLoader is not None:
        print(validation(model, testLoader, transform))  
    
    return model

def periodical_trainer(model:Module,trainloader:DataLoader,epochs_to_trainfor:int,transform:Any,filepath:str,\
                        epoch_period:int=None,singleTransformation:bool=False)->Module:
    r"""
        Will train a model and save the various parameters that go with the training
        at either every epoch or every specified number of epochs. This method requires
        a filepath to be sent in and the functionalities of the method will be based
        on whether the file with the filepath exists.

        The file will include the following stuff:
            epoch: the number of epochs that have been trained already
            model_parameters: the saved parameter for the model
            optimizer_parameters: saved parameters for optimizers
        
        -------------------
        params:
            - model: model loaded from torchvision.models to be used for training
            - trainLoader: training dataset
            - epochs_to_trainfor: how many iterations to train the model for
            - transform: defensive transformation to be applied for training
            - filepath: path to file where model parameters will be saved during/after training
            - epoch_period: number of epochs to train for before recording stats and saving current model parameters
            - singleTransformation: If True then transformation is done one image at a time using singleImageTransformForBatch() method.
        
        ------------------
        return:
            - trained model

    """

    model.to(device)
    ce_loss = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = optim.Adam(params=params,lr=.0001)
    epoch = 0
    loss = 0


    #read the file if it exists and get pre-saved parameters:
    if path.exists(filepath):
        # get saved dictionary:
        training_checkpoint = torch.load(filepath)
    
        #extract the different parts of the dictionary for use
        modelparams = training_checkpoint[MODEL_PARAMETERS]
        optimparams = training_checkpoint[OPTIMIZER_PARAMETERS]
        epoch = training_checkpoint[EPOCH_COUNT]
        loss = training_checkpoint[LOSS]
    
        #set the pre-saved values:
        model.load_state_dict(modelparams)
        optimizer.load_state_dict(optimparams)

  
    totalTimeElapsed = 0
    numBatchesToTrack = 10
    timeDiff_batch = 0
    # do training:
    while epoch < epochs_to_trainfor:
        startTime = time.time()
        batchCounter = 0
        for imgs, lbls in trainloader:
            startTime_batch = time.time()
            image = imgs.to(device)
            label = lbls.to(device)
          
            if singleTransformation:
                trans_img = singleImageTransformForBatch(image,transform)
            else:
                trans_img = transform(image)

            output = model(trans_img)
            model.zero_grad()
            loss = ce_loss(output,label)
            loss.backward()
          
            optimizer.step()
            batchCounter += 1
            endTime_batch = time.time()
            timeDiff_batch += endTime_batch - startTime_batch
            timeDiff_batch_formatted = formatTimeDiff(timeDiff_batch)
            if batchCounter % numBatchesToTrack == 0 or batchCounter == 1:
                print(f"Batch {batchCounter} completed. Required {timeDiff_batch_formatted}")
                timeDiff_batch = 0 if batchCounter != 1 else timeDiff_batch
    
        #update epoch and see if its time to break loop yet; also print details
        epoch += 1
        endTime = time.time()
        timeDiff = endTime - startTime
        totalTimeElapsed += timeDiff
        formattedTimeDiff = formatTimeDiff(timeDiff)
        formattedTotalTime = formatTimeDiff(totalTimeElapsed)
        print(f"Epoch completed: {epoch}. With loss: {loss}. Epoch Time: {formattedTimeDiff}. Total time elapsed: {formattedTotalTime}")
        if epoch % epoch_period == 0: # did number of epochs specified in epoch period
            break
  
    # create dictionary that will be saved to file using same keys:
    toBeSaved = {}
    toBeSaved[MODEL_PARAMETERS] = model.state_dict()
    toBeSaved[OPTIMIZER_PARAMETERS] = optimizer.state_dict()
    toBeSaved[EPOCH_COUNT] = epoch
    toBeSaved[LOSS] = loss
    torch.save(toBeSaved,filepath)
  
    return model

def validation(model:Module, testloader:DataLoader, transformer:Any)->float:
    r"""
        Does validation run using test dataset.
    """
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader: 
            images = images.to(device)
            labels = labels.to(device)
            trans_img = transformer(images)
            outputs = model(trans_img)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total