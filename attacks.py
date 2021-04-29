"""
    Adversarial attacks implemented by Capstone Group 27.

    Each attack is implemented as a function. These can be called
    in a standalone manner or from the driver file associated with
    this project.
"""

# imports:
from typing import Tuple
import numpy as np
import torch
from torch import optim

# global variables:
device = 'cuda:0' if torch.cuda.is_available else 'cpu' # set gpu if its available


# Attacks:
"""
    Capstone Group 27's implementation of the Carlini Wagner Attack from (N. Carlini, D. Wagner. “Towards evaluating the robustness of neural networks”. arXiv:1608.04644 [cs.CR])
    based on the experimental setup from (C. Guo, M. Rana, M. Cisse and L Maaten,  “Countering adversarial images using input transformations,” International Conference on Learning Representations, 2018.).
    This is a simpler and altered implementaion relative to the code provided by the authors. 

    This code as several key differences from the works that it was based on:
        - This implementation is done in PyTorch (instead of TensoFlow) and only uses the untargeted version of the attack. 
        - Also this code only implements an attack instead of accounting for specific defenses or carrying out image 
          processing as one might see in the code provided by authors of “Countering adversarial images using input transformations”. 
        - Furthermore, this code utilizes a method to convert target labels to one hot encoding. The one-hot encoding serves a search 
          purpose and is only used for that. 
        - Another difference that can be found in this code is that images are not normalized to be between 0 and 1 for the most part. 
          Normalization is done to check for L2 Dissimilarity, see (c. Guo et al.). This code utilizes CIFAR10 images as they are.
        - This code has its own version of adjusting for L2 Dissimilarity due to lack of details in (C. Guo et al.).
  
    This method implements L2 Carlini Wagner attack with hinge loss outlined in (N. Carlini, D. Wagner). Optimization of the following function is implemented:
                || ((1/2) tanh(w) + 1) - x ||(2,2)  + (const) max(score of true label - score of greatest non-true label, k)
  
    This function and the one that will be implemented below is slightly different that the equation written in the paper, however it follows (C. Guo et al.) and
    some code based adjustments found in the code for (N. Carlini, D. Wagner). Based on (C. Guo et al.):
        - the const was set to and kept at 10
        - the final perturbation was adjusted for particular L2 Dissimilarities

"""
def cwl2atk(model:torch.nn.Module,imgs:torch.Tensor,targets:torch.Tensor,e:float=0.0,numIterations:int=50,boxAdjustable:float=0.5,const:int=10,\
            k:int=0,learningRate:float=0.001,train:bool=False,breakThreshold:any = None,minIterations:int=20)->Tuple:

    r"""
        Method generates Carlini Wagner Attack (N. Carlini, D. Wagner) images given a model, data, and other parameters. 
    
        Note in this method "w" will be the difference between the image in tan space and the attack image. The attack image
        will be the sum of the image in tan space and the modifier variable "m"; this modifier variable will be the variable that
        will be optimized.

        --------------------
        @params:
            - imgs: Images to obtain perturbations for. should be four dimensional input (batch X channels X height X width).
                    If the batch size is 1 then "train" parameter should be set to False.
            - targets : The true labels that go with the images. Dim = (batch x 1)
            - model : The model that the attack is being run on. Should be set to proper device (cuda or cpu) before being sent to this method.
            - e :  The L2 dissimilarity factor that has to be set for experimentation. This will be used to determine the perturbation multiplier.
                   See (C. Guo et al.)
            - numIterations : the number of iterations to run the attack for.
            - boxAdjustable : multiplier that is used in  || ((1/2) tanh(w) + 1) - x ||(2,2). The (1/2) is adjustable based on what was found in the code for
                             (N. Carlini, D. Wagner) but this code uses (1/2).
            - const : the constant by which hinge loss is multiplied; lambda in the (C. Guo et al.) paper. The value of 10 is used for all instances.
            - k : see k in (N. Carlini, D. Wagner); kept at 0 at all times
            - learningRate : learning rate for optimizer; the default is 0.001.
            - train : determines what mode the model should be set to. If True then model is set to .train() otherwise .eval(). Note if batch size is = 1 then
                      this parameter has to be False or there will be errors.
            - breakThreshold : determines, along with minIterations, when a loop should break. If after minIterations this number of images
                               are still classified correctly then loop can break.
            - minIterations : how many iterations to wait for until checking if the loop should break.
    

        -------------------
        @returns:
            attackImages : attack images with perturbations added.
            perturbation : the perturbation generated by this method; m in this method
    """
    
    # sub-methods 
  
    def convertToOneHot(labels:torch.Tensor,numClasses:int=10,mask:bool=False)->torch.Tensor:
        r"""
            Gets all of the labels for this batch (batch size x 1) then converts it to
            one-hot (batch size x number of classes = 10). 
        """
        tensorType = torch.bool if mask else torch.float
        encodeVal = True if mask else False
        encodedTensor = torch.zeros(len(labels),numClasses,dtype=tensorType,device=device)
        for i in range(encodedTensor.size()[0]):
            newTensor = torch.zeros((1,10),dtype=tensorType,device=device)
            newTensor[0,labels[i]] = encodeVal
            encodedTensor[i] = newTensor
        return encodedTensor
  
    def getScores(scores:torch.Tensor,targets:torch.Tensor,numClasses:int=10)->Tuple:
        r"""
            Sub-routine for the Carlini Wagner Attack (N. Carlini, D. Wagner). This method obtains both
            the score and label of the true class for a particular image. This method also gets the
            score and the index of the greatest non-true label for an image.
        """

        targetMask = convertToOneHot(targets,numClasses=numClasses,mask=True)
  
        scores_forNonTrueTarget = scores.clone().detach()
        scores_forNonTrueTarget[targetMask] = -10000 #there will only be one True value in the target mask
        nonTrueMaxScore = scores_forNonTrueTarget.max(1)

        trueMaxScore = (scores[targetMask],targets)

        return trueMaxScore,nonTrueMaxScore #these are both tuples that include the indicies of the max scores along with the scores themselves
  
  
    #############################################################
    #############################################################
  
  
    # Main Method:

    # - variable declarations - :
    #generic vars:
    batchSize = len(imgs)
  
    imgs = imgs.to(device)
    imgs_rangeSet = _setRange(imgs)
    targets = targets.to(device)

    #imgs_tanSpace = torch.tanh(imgs*boxAdjustable) #put image to tan space for the  || ((1/2) tanh(w) + 1) - x ||(2,2) calculations
    imgs_tanSpace = torch.tanh(imgs_rangeSet)
  
    model = model.to(device)
    if train:
        model.train()
    else:
        model.eval()
  
  
    const = torch.ones(batchSize,device=device,dtype=torch.float) * const
  
    m = torch.ones(imgs.size(),device=device) * 1e-3 # the perturbations that will be added to the image to generate the attack image; this will be optimized
    m.requires_grad=True

    optimizer = optim.Adam([m],lr=learningRate)

    #best vars: (used of optimizing the method; may not actually be used)
    bestL2 = torch.ones(batchSize,dtype=torch.float,device=device) * 1e10
    bestScores = torch.ones(batchSize,dtype=torch.float,device=device) * -1
    bestM = torch.zeros(imgs.size(),dtype=torch.float,device=device)


    # - run attack iterations - :
    for i in range(numIterations):
        with torch.set_grad_enabled(True):
            ##--- part 1: run single attack

            ### --generate atkImg-- :
            atkImgs = (imgs_tanSpace + torch.tanh(m)) * boxAdjustable

      
            ### -- generate loss function -- :
      
            l2Dist = torch.sum(((imgs_tanSpace - atkImgs)**2),(1,2,3)) ** 0.5

            # generate the hinge loss portion of the loss function: max(score for true label - max(score for non-true label), k)
            #--- first get the model results with the atk images, then extract the necessary scores from the results, finally do hinge loss calc
            modelPred = model(imgs + (m*255))
            currentPreds = torch.max(modelPred,1)[1]
            real,atk = getScores(modelPred,targets)
            realLabelScore,realLabel = real
            atkScores,atkLabels = atk
            hingeLoss = torch.max(torch.tensor([k],device=device),realLabelScore-atkScores)
      
            # get total values:
            loss_1half = torch.sum(l2Dist)
            loss_2half = torch.sum(hingeLoss*const)
            totalLoss = loss_1half + loss_2half
      
      
            ### -- do optimization-- :
            model.zero_grad()
            totalLoss.backward()
            optimizer.step()

      
            ## --- part 2: update best running values --- :
            # (may not actually be used)
            if i == 0:
                bestM = m.clone().detach()
                bestL2 = l2Dist.clone().detach()
            else:
                #create mask for best values:
                diffLabelMask = currentPreds != targets #mask for outputs from the model being different from the true label of the images.
                lessL2Mask = bestL2 <= l2Dist
                bestMask = diffLabelMask * lessL2Mask
                bestM[bestMask] = m[bestMask]
      

            ## --- part 3: check if loop should be broken --- : 
            # broken if 5 or less images are being predicted correctly and we are past 20 minIterations
            numCorrects_inner = torch.sum(currentPreds == targets)
      
            if breakThreshold is not None:
                if numCorrects_inner <= breakThreshold and i>minIterations:
                    break

      
  
    # - Final Adjustments and Return - :
    multiplier = _adjustMForL2Dissim(imgs, bestM * 255,e) # carlini wagner attack l2 disimilarity adjustment
    return imgs + bestM * multiplier * 255,bestM * multiplier * 255


"""
    Capstone Group 27's implementation of DeepFool from “DeepFool: a simple and accurate method to fool deep neural networks” by S.-M. Moosavi-Dezfooli, A. Fawzi, P. Frossard.
    Our specific code was based on code from https://github.com/ej0cl6/pytorch-adversarial-examples/blob/master/attackers.py .  
"""
def DeepFool(model:torch.nn.Module, x:torch.Tensor, y:torch.Tensor, e:float=0.0, max_iter:int=5, clip_max:float=0.5, clip_min:float=-0.5)->torch.Tensor:
    r"""
        DeepFool adversarial attack implementation.

        -----------------
        @params:
            - e = dissimilarity factors
            - mat_iter = maximum number of iterations used for the attack; time required for attack increases with this value
        
        -----------------
        Note: This method/attack is a single image operation. It requires batch size to be 1 (1 x c x w x h) or only a single
        image input.

        -----------------
        @returns:
            - tensor - adversarial images generated by this method
    """

    model.eval()
    if len(x.size()) == 3: # 3-dimesional input
        nx = torch.unsqueeze(x,0)
    else:
        nx = x.clone().detach()
    nx.requires_grad_()
    eta = torch.zeros(nx.shape, device= device)

    out = model(nx+eta)
    n_class = out.shape[1]
    py = out.max(1)[1].item()
    ny = out.max(1)[1].item()

    i_iter = 0

    while py == ny and i_iter < max_iter:
        out[0, py].backward(retain_graph=True)
        grad_np = nx.grad.data.clone()
        value_l = np.inf
        ri = None

        for i in range(n_class):
            if i == py:
                continue

            nx.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = nx.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, py]
            value_i = np.abs(fi.item()) / np.linalg.norm(wi.cpu().numpy().flatten())

            if value_i < value_l:
                ri = value_i/np.linalg.norm(wi.cpu().numpy().flatten()) * wi

        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        py = out.max(1)[1].item()
        i_iter += 1
        
    multiplier = _adjustMForL2Dissim(nx,eta,e)
    x_adv = nx + eta * multiplier
    x_adv.squeeze_(0)
        
    return x_adv.detach()


"""
    Capstone Group 27's implementation of Fast Gradient Sign Method from “Explaining and harnessing adversarial examples” by I. Goodfellow, J. Shlens, and C. Szegedy.
"""
def FGSM_Attack(model:torch.nn.Module, image:torch.Tensor, target:torch.Tensor, e:float=0.007)->torch.Tensor:
    r"""
        Fast Gradient Sign Method implementation. This attack does not require additional processing for L2 Dissimilarity
        factors. The desired L2 Dissimilarity factor can be obtained by simply multiplying the adversarial perturbation
        by the desired factor.
        --------------
        @params:
            - model : model used for attack
            - image : batch of images to generated adversarial counterparts for
            - target: tensor of true class labels for the images
            - e: desired dissimilarity factor
        --------------
        @returns:
            - tensor - adversarial images
    """

    model.eval()

    image.requires_grad = True

    modelOutput = model(image)

    loss = torch.nn.functional.cross_entropy(modelOutput,target)
    model.zero_grad()
    loss.backward()
    img_grad = image.grad.data
  
    sign_img_grad = img_grad.sign()
  
    atkImg = image.clone().detach() + e*sign_img_grad

    return atkImg


"""
    Capstone Group 27's implementation of Iterative Fast Gradient Sign Method from “Adversarial machine learning at scale” by A. Kurakin, I. Goodfellow, S. Bengio.
"""
def IFGSM_Attack(model:torch.nn.Module,image:torch.Tensor,target:torch.Tensor,e:float=0.007,num_iterations:int=10,clipVal:float=0.08)->torch.Tensor:
    r"""
        Iterative Fast Gradient Sign Method (IFGSM) implementation. This attack does not require additional L2 Dissimilarity
        processing. The desired L2 Dissimilarity factor can be achieved by multiplying the perturbation by the desired
        L2 dissimilarity factor passed in.
        ----------------
        params:
            - model : model used for attack
            - image : batch of images to generated adversarial counterparts for
            - target: tensor of true class labels for the images
            - e : desired L2 Dissimilarity factor
            - num_iterations : number of iterations to run the FGSM method for
            - clipVal : restrict the size of perturbations to range defined by (-clipVal,clipVal)
        ----------------
        returns:
            - tensor : adversarial images
    """
    image = image.to(device)
    target = target.to(device)
    model.to(device)
    model.eval()

    image.requires_grad = True

    diff = torch.zeros(image.size(),device=device)
    for i in range(num_iterations):
        model.zero_grad()
        output = model(image + diff)
        loss = torch.nn.functional.cross_entropy(output,target)
        loss.backward()
        signImgGrad = image.grad.data.sign()
        diff += e * signImgGrad
        diff = diff.clamp(-clipVal,clipVal) # may need to revisit this
  
    atkImg = image + diff
    return atkImg



# Attack Utilities
"""
    Common methods to be used by each of the attacks. They primarily have to do with calculating the
    L2 Dissimilarity return an appropriate multiplier.
"""

def _adjustMForL2Dissim(origImages:torch.Tensor,perturbation:torch.Tensor,e:float)->torch.Tensor:
    r"""
        Method to check the Normalized L2 Dissimilarity between an image and its adversarial
        counterpart and then determine the proper factor to obtain the desired dissimilarity
        factor value e.
        
        ---------------
        e : desired dissimilarity factor

        ---------------
        See “Countering adversarial images using input transformations” by C.Guo et.al. for more details on
        the operation. Our implementation differs in that the images are not normalized between 0 and 1, this
        is done through the setRange() method.
    """
    origImgs = origImages.clone().detach()
    atkImgs = origImgs + perturbation
    origImgsNormalized = _setRange(origImgs)
    atkImgsNormalized = _setRange(atkImgs)

    ## calculate l2 dissimilarity:
    origL2 = torch.sum(origImgsNormalized**2,(1,2,3)) ** 0.5
    diffL2 = torch.sum((origImgsNormalized-atkImgsNormalized)**2,(1,2,3)) ** 0.5

    multiplier = (e*len(origImages))/(torch.sum(diffL2/origL2))
    return multiplier

def _setRange(imageBatch:torch.Tensor)->torch.Tensor:
    r"""
        Sets the range for the images in the batch to [0,1], normalizing the images. The function
        output = (original - minimum) / (maximum - minimum) is used. Min/max values are found
        using the getMinMax() method.
    """
    minT,maxT = _getMinMax(imageBatch)
    rangeSetTensor = (imageBatch - minT) / (maxT - minT)
    return rangeSetTensor

def _getMinMax(imageBatch:torch.Tensor)->Tuple:
    r"""
        Gets the max and min pixel values per image and returns those values as
        tensors matching the dimensions of imageBatch.

        -----------------
        params:
            - imageBatch: batch of images
        
        -----------------
        returns:
            tuple:
                0 - minvals:torch.Tensor - tensor with the minimum pixel value for each image
                1 - maxvals:torch.Tensor - tensor with the maximum pixel value for each image
    """
    imgSize = imageBatch.size()
    minVals = torch.ones(imgSize,device=device)
    maxVals = torch.ones(imgSize,device=device)
    for i in range(len(imageBatch)):
        currentImg = imageBatch[i]
        currentMax = torch.max(currentImg)
        currentMin = torch.min(currentImg)
        minVals[i] *= currentMin
        maxVals[i] *= currentMax
    return minVals,maxVals
  


