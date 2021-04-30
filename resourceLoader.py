"""
    Loads pre-trained models and datasets. Methods in this file require
    constantds which will be made available by this file. These constants
    will have be imported into the calling file.
"""

# imports
from typing import Any,Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision import transforms


# Constants:
ID_MNIST = 0
ID_CIFAR10 = 1

T_CIFAR_FOR_MODEL = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TMnist = transforms.Compose([
      transforms.Grayscale(num_output_channels=3),
      transforms.ToTensor()
])

T_Basic_CIFAR = transforms.Compose([transforms.ToTensor()])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

####################################################



# Model and Data Retrieval

def getModelandDataset(modelParamPath:str,dataId:int,transformer:Any,batchSize:64,train:bool=False)->Tuple[torch.nn.Module,DataLoader]:
    r"""
        Will use a resnet18 model with 10 output classes. Must pass in path to file with pre-trained model data. A resnet18 model
        should be prepared in advanced with the desired defensive transformation.
        ------------------------
        @params:
            - modelParamPath: path to the pre-trained model parameters
            - dataId: id for the dataset to be loaded, the constants for this parameters can be imported from the resourceLoader file
            - transformer: defensive transformation to be applied to dataset, they can be imported from resourceLoader (this) file
            - batchSize: size of each batch in the dataset
            - train: whether the training dataset should be loaded or the test
        ------------------------
        @returns:
            - Tuple:
                0 - Loaded Model
                1 - Dataset
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512,10)
    model = model.to(device)

    model,dataLoader = _loadModelAndDataset(model,modelParamPath,dataId,transformer,batchSize,train)

    return model,dataLoader

def _loadModelAndDataset(model:torch.nn.Module,modelPath:str,dataId:int,transformer:Any,batchSize:int,train:bool=False)->Tuple[torch.nn.Module,DataLoader]:
    r"""
        Helper method to load the pre-trained model state dictionary and dataset/dataloader. This function has a model parameter
        that can accept any model. Being able to accept any model is different from the exposed getModelandDataset() model which
        will only load the ResNet18 model. getModelandDataset() calls this method with the ResNet18 model, this method can also
        be used in a standalone method to work with any other model.
    """
    if modelPath == '':
        dataLoader = None
        if dataId == ID_MNIST:
            dataLoader = DataLoader(datasets.MNIST('/content',download=True,transform=transformer,train=train),batch_size=batchSize,shuffle=True)
        elif dataId == ID_CIFAR10:
            dataLoader = DataLoader(datasets.CIFAR10(root='./data', train=train, download=True, transform=transformer),batch_size=batchSize,shuffle=True)
        return model,dataLoader

    #load model params:
    if dataId == ID_MNIST:
        model.load_state_dict(torch.load(modelPath)['model_state_dict'])
        dataLoader = DataLoader(datasets.MNIST('/content',download=True,transform=transformer,train=train),batch_size=batchSize,shuffle=True)
  
    if dataId == ID_CIFAR10:
        model.load_state_dict(torch.load(modelPath))
        dataLoader = DataLoader(datasets.CIFAR10(root='./data', train=train, download=True, transform=transformer),batch_size=batchSize,shuffle=True)
  
    return model,dataLoader

  