"""
    IO methods used for the project. Primarily involves plotting, recording, and reading data obtained from 
    tests. Best used in conjuntion with this project's driver but can be used in a standalone manner.
"""

#imports:
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import glob
from os import path

#constants
LINE_COLOR = "color"
LINE_LABEL = "label"
LINE_DATA = "data"
LINE_XRANGE = "xrange"
DATA_FILE_PATH = "./*_graphData.txt"


# Interface Methods:

## Plotting Utility Methods
"""
    Methods that use Matplotlib to plot the data obtained from testing the model with various attacks and defenses. These methods
    are also responsible for writing the data obtained from testing into text files and then reading them.
"""

def plot_acc(accuracies:List[float],xRange:List[float]=None):
    plt.figure()
    plt.clf()
    plt.title('Accuracies')
    plt.ylabel('Accuracy')
    print("Mean:", np.mean(accuracies))
    if xRange is None:
        plt.plot(accuracies)
    else:
        plt.plot(xRange,accuracies)
    plt.show()

def generateCumulativeGraph(filePaths:str = DATA_FILE_PATH,gTitle:str='',xRange:List[float] = None,singleSet:bool = False,genSingleSetTxt:str = '',\
                            altColors:bool=False,altPaletteFilePath:str='./dimmed_palette.txt'):
    r"""
    """

    xRange = np.arange(0,0.08,0.005) if xRange is None else xRange
    graphInfoList = _read_data_files(filePaths) if not singleSet else [_read_data_files_singleset(filePaths)]
    altColorDict = _read_alt_palette_file(altPaletteFilePath) if altColors else {}
  
    graphTitle = 'Accuracies' if gTitle == '' else 'Accuracies: ' + gTitle

    plt.figure()
    plt.title(graphTitle)
    plt.ylabel('Accuracy')
    plt.xlabel('Dissimilarity')

    for graphInfo in graphInfoList:
        #non key word arguments
        lData = graphInfo[LINE_DATA]
        if singleSet:
            xRange = graphInfo[LINE_XRANGE]
        # additional/keyword arguments:
        additionalArgs = {}
        lLabel = graphInfo[LINE_LABEL]
        lColor = graphInfo[LINE_COLOR]
        additionalArgs[LINE_LABEL] = lLabel
        additionalArgs[LINE_COLOR] = graphInfo[LINE_COLOR]
        if altColors:
            styleDict = altColorDict[lLabel]
            for key,value in styleDict.items():
                additionalArgs[key] = value
        plt.plot(xRange,lData,**additionalArgs)
  
    plt.legend()

    if genSingleSetTxt != '':
        lData = lData.tolist() if isinstance(lData,np.ndarray) else lData
        write_graph_data(lData,lLabel,lColor,genSingleSetTxt+lLabel,xRange)

## writing output interface method
def write_graph_data(data:List[float],label:str,color:str,fileName:str,x_range:List[float]=None,seqNum:int=None):
    r"""
        Writes the details of the graph like the actual data, the line's color,
        and its label into a text file. The format of the file is pre-set and can be
        found in an example "*_graphData.txt" file.

        ------------------
        The x_range is by default None because it will follow our usual x_range of 
        np.arange(0,0.08,0.005). Currently there is no code to process any other x_range
        aside from the default.
    """

    # check if fileName is proper for our purposes:

    if seqNum is not None:
        if fileName.endswith('.txt'):
            fileName = fileName[:len(fileName)-4]
        fileName += '_' + str(seqNum) + '.txt'

    if not fileName.endswith("_graphData.txt"):
        if fileName.endswith('.txt'):
            fileName = fileName[:len(fileName)-4]
        fileName+= "_graphData.txt"

    color_line = LINE_COLOR + " " + color
    label_line = LINE_LABEL + " " + label
    data_line = LINE_DATA + " " + str(data)
    lineList = [color_line,label_line,data_line]
    with open(fileName,'w') as dest:
        for l in lineList:
            dest.write(l + "\n")
            print(l)
        if x_range is not None:
            x_range = x_range.tolist() if isinstance(x_range,np.ndarray) else x_range
            l = LINE_XRANGE + " " + str(x_range) + "\n"
            dest.write(l)
            print(l)

####################################################



# Utility Functions for reading input data
def _read_data_files(path:str = DATA_FILE_PATH):
    r"""
    """
    graphDict_list = []
    filePaths = glob.glob(path)
    for singlePath in filePaths:
        graphDict = {}
        with open(singlePath,'r') as fileToBeRead:
            fileLines = fileToBeRead.readlines()
            for line in fileLines:
                line_tokenized = line.split(maxsplit=1)
                key = line_tokenized[0]
                if key == LINE_DATA or key == LINE_XRANGE:
                    graphComponent = _read_np_array(line_tokenized[1])
                else:
                    graphComponent = line_tokenized[1].strip('\n')
                graphDict[key] = graphComponent
        graphDict_list.append(graphDict)
  
    return graphDict_list

def _read_data_files_singleset(path:str = DATA_FILE_PATH):
    r"""
        Used for reading different parts of the same set of data. Used when different
        epochs are run on different session and need to aggregate the data.
    """

    graphDict = {}
    filePaths = glob.glob(path)
    filePaths = sorted(filePaths) # sort so that earlier data is first
    counter = 0
    for singlePath in filePaths:
        if counter == 0: # set up the first bits of data
           with open(singlePath,'r') as fileToBeRead:
              fileLines = fileToBeRead.readlines()
              for line in fileLines:
                line_tokenized = line.split(maxsplit=1)
                key = line_tokenized[0]
                if key == LINE_DATA or key == LINE_XRANGE:
                    graphComponent = _read_np_array(line_tokenized[1])
                else:
                    graphComponent = line_tokenized[1].strip('\n')
                graphDict[key] = graphComponent
        else: # append the x_range data and the data data to existing items:
            with open(singlePath,'r') as fileToBeRead:
                fileLines = fileToBeRead.readlines()
                for line in fileLines:
                    line_tokenized = line.split(maxsplit=1)
                    key = line_tokenized[0]
                    if key != LINE_DATA and key != LINE_XRANGE:
                        continue
                    currentArr = graphDict[key]
                    arrToBeAdded = _read_np_array(line_tokenized[1])
                    graphDict[key] = np.concatenate((currentArr,arrToBeAdded))
        counter += 1
  
    return graphDict

def _read_np_array(dataLine:str):
    r"""
        Reads an array from a text file and returns a numpy array with the data.
    """
    arr_data = dataLine.strip('\n')
    arr_data = arr_data.strip('[').strip(']').split(',')
    valList = []
    for elem in arr_data:
        valList.append(float(elem))
    valList = np.array(valList)
    return valList

def _read_alt_palette_file(filePath:str='./dimmed_palette.txt'):
    r"""
        Read the alt palette file. The alt palette file contains the altered
        colors for each defense. Altered colors for clean_model,crop_rescale, and
        bitdepth_reduction are used to highlight performance.
    """
  
    if filePath is None or not path.exists(filePath):
        return {}
  
    newColorDict = {}

    with open(filePath,'r') as altfile:
        filelines = altfile.readlines()
        for line in filelines:
            line_arr = line.split()
            graphName = line_arr.pop(0)
            styleDict = {}
        for elem in line_arr:
            style_arr = elem.split('-')
            styleDict[style_arr[0]] = style_arr[1]
        newColorDict[graphName] = styleDict
  
    return newColorDict

####################################################
