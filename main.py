"""
    Examples of how our test procedure is run.
"""

from resourceLoader import getModelandDataset,ID_CIFAR10,T_CIFAR_FOR_MODEL
from attacks import cwl2atk
from defenses import T_CROP_RESCALE
from drivers import runTest

"""
    For each test we have to do some setup and resource gathering before the tests are run. First, the
    pretrained model and dataset have to be loaded. Then the desired range of L2 Dissimilarity has to be
    set either by manually creating a list or using numpy.
"""

if __name__ == "__main__":
    #running carlini wagner attack on crop-rescale
    batchSize = 64
    eRange = [0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075]
    model,testLoader = getModelandDataset('./modelpath.pth',ID_CIFAR10,T_CIFAR_FOR_MODEL,batchSize=batchSize,train=False)
    runTest(cwl2atk,model,testLoader,eRange=eRange,defense=T_CROP_RESCALE,writeToFile=True,filePath='outputfile.txt',\
            graphLabel='crop-rescale',graphColor='red')

