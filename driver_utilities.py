"""
    Utility functions that are used by both train and test drivers.
"""



#imports
from typing import Any
import torch



#constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# methods
def formatTimeDiff(timeDiff:int)->str:
    r"""
        Formats the time from seconds into hh:mm:ss format.
    """
    seconds_in_hour = 60*60
    seconds_in_minute = 60

    hours_in_time = int(timeDiff // seconds_in_hour)
    timeDiff -= hours_in_time * seconds_in_hour
    minutes_in_time = int(timeDiff // seconds_in_minute)
    timeDiff -= minutes_in_time * seconds_in_minute
    seconds_in_time = int(timeDiff)

    timeDiffFormatted = f"{hours_in_time}h::{minutes_in_time}m::{seconds_in_time}s"
    return timeDiffFormatted

def singleImageTransformForBatch(imageBatch:torch.Tensor,transformation:Any,reqModel:bool=False,\
                model:torch.nn.Module=None,targets:torch.Tensor=None,**kwargs)->torch.Tensor:
    r"""
        Applies transformation to each image in the batch. The transformation can be defensive or offensive.

        ------------------------
        @param:
            - imageBatch: batch of images; 4-d input
            - transformation: method that the images will be inputted into; pass in the method name

        Following parameters are only for attack transformations that use the model:
            - reqModel: the transformation requires the use of a machine learning model; if True pass in model and targets and possibly kwargs
            - model: model being used, only use when needed
            - targets: targets that correspond with the images, only use when needed
            - kwargs: additional (named) parameters that are required for the model or transformation; 
                      check the model/transformation for details on these
    
        ------------------------
        @returns:
            - batch of transformed images
    """

    toBeReturned = torch.zeros(imageBatch.size(),device=device)
    for i in range(len(toBeReturned)):
        if not reqModel: # probably defensive transformation
            toBeReturned[i] = transformation(imageBatch[i],**kwargs)
        else:
            toBeReturned[i] = transformation(model,imageBatch[i],targets[i],**kwargs)
    return toBeReturned

