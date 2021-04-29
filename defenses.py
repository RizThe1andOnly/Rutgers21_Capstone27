"""
    Defenses used for the Capstone group 27's experiments and analysis. These are all
    objects of the Pytorch 'transforms' class. They will be passed into either Dataloaders
    or model testing driver methods.
"""

from torchvision import transforms

# defensive transforms for:
# - crop rescale
# - bit depth reduction

h = 224
w = 224
T_CROP_RESCALE = transforms.Compose([
    transforms.RandomCrop((round(h*3/4), round(w*3/4))), 
    transforms.Resize((h,w))
    ])

T_BITDEPTH_REDUCTION = transforms.Grayscale(num_output_channels=3)