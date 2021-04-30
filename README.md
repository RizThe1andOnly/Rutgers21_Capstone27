# Rutgers 2021 Capstone Project Group 27
## Electrical and Computer Engineering
----------------------------------------------

A research project based on adversarial machine learning. The project involves recreating and reassesing experiments
that test various adversarial attacks against defenses. This repository includes re-created codes for several
famous adversarial attacks and input transformations for defenses.

----------------------------------------------

### Authors

- Neel Amin
- Netanel Arussy
- Rizwan Chowdhury
- Talya Kornbluth

----------------------------------------------

### Contents

- Attacks and defenses are defined in the correlating files.
- train_drivers.py contains code to retrain the models obtained from torchvision.models
- test_drivers.py contains the code that runs tests on models with the given adversarial and defensive transformations.
- driver_utilities.py has common methods used by train and test drivers.
- io_methods.py has methods that show figures and write output data to text files.
- sampling.py has methods that display images with and without adversarial transformations.
- resourceLoader.py obtains the specified dataset and model (from those available in torchvision.datasets and torchvision.models respectfully)
- main.py has an example on how to run the tests for the project.

----------------------------------------------

### References
[1] C. Guo, M. Rana, M. Cisse and L Maaten,  “Countering adversarial images using input transformations,” International Conference on Learning Representations, 2018.

[2] I. Goodfellow, J. Shlens, and C. Szegedy. “Explaining and harnessing adversarial examples,” International Conference on Learning Representation, 2015. arXiv:1412.6572  

[3] S.-M. Moosavi-Dezfooli, A. Fawzi, P. Frossard. “DeepFool: a simple and accurate method to fool deep neural networks,” Ecole Polytechnique Fédérale de Lausanne, 2016. URL: arXiv:1511.04599

[4] A. Kurakin, I. Goodfellow, S. Bengio. “Adversarial machine learning at scale,”  International Conference on Learning Representations, 2017.

[5] N. Carlini, D. Wagner. “Towards evaluating the robustness of neural networks”. arXiv:1608.04644 [cs.CR] 

[6] K. HE, X. ZHANG, S. REN, and J. SUN. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016), pp. 770–778. arXiv:1512.03385 [cs.CV]

[7] W. Xu, D. Evans, Y. Qi. “Feature squeezing: detecting adversarial examples in deep neural networks,” Network and Distributed Systems Security Symposium, 2018. arXiv:1704.01155 [cs.CV].

[8] G. K. Dziugaite, Z. Ghahramani, and D. Roy. “A study of the effect of JPG compression on adversarial images,” CoRR, abs/1608.00853, 2016.

[9] L. Rudin, S. Osher, and E. Fatemi. “Nonlinear total variation based noise removal algorithms,” Physica D, 60:259–268, 1992.

[10] A. Efros and W. Freeman. “Image quilting for texture synthesis and transfer,” In Proc. SIGGRAPH, pp. 341–346, 2001.

[11] Y. Lecun, C. Cortes, and C. J. Burges. “The mnist database of handwritten digits,” 1998.

[12] A. Krizhevsky, and G. Hinton. “Learning multiple layers of features from tiny images”.


