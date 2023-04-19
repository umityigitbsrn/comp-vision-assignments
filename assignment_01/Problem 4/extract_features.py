import os
import numpy as np
import scipy.io as sio
import glob
import torchvision
from models.alexnet import alexnet
from models.vgg import vgg16
import torchvision.transforms as transforms
from PIL import Image

def main():

  alex_feature = []
  alex_label = []
  
  vgg16_feature = []
  vgg16_label = []

  transform  = transforms.Compose([
    transforms.Scale(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

  train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

  test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

  
  # [Problem 4 a.] IMPORT VGG16 AND ALEXNET FROM THE MODELS FOLDER WITH 
  # PRETRAINED = TRUE

  vgg16_extractor = 
  vgg16_extractor.eval()
  
  alex_extractor = 
  alex_extractor.eval()
  
  for idx, data in enumerate(train_data):

      image, label = data
      
      # [Problem 4 a.] OUTPUT VARIABLE F_vgg and F_alex EXPECTED TO BE THE 
      # FEATURE OF THE IMAGE OF DIMENSION (4096,) AND (256,), RESPECTIVELY.
      F_vgg = 
     
      vgg16_feature.append(F_vgg)
      vgg16_label.append(c)
    
      F_alex = 
      alex_feature.append(F_alex)
      alex_label.append(c)

  sio.savemat('vgg16.mat', mdict={'feature': feature, 'label': label})
  sio.savemat('alexnet.mat', mdict={'feature': feature, 'label': label})



def KNN_test(train_mat_file, test_data, K = 1):	
    # FILL IN TO LOAD THE SAVED .MAT FILE
    vgg_mat =
    alex_mat =

    vgg16_extractor =
    alex_extractor = 

    for idx, data in enumerate(test_data):

        # 1. # EXTRACT FEATURES USING THE MODELS - ALEXNET AND VGG16
        F_test_vgg16 = 
        F_test_alex = 

        # 2. # FIND NEAREST NEIGHBOUT OF THIS FEATURE FROM FEATURES STORED IN ALEXNET.MAT AND VGG16.MAT
        
        # 3. # COMPUTE ACCURACY	
        alex_accuracy = 0.0
        vgg16_accuracy = 0.0

        return vgg16_accuracy, alex_accuracy
	
if __name__ == "__main__":
   main()
