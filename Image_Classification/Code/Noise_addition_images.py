
# coding: utf-8

# In[31]:

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os          
from tqdm import tqdm
import matplotlib.pyplot as plt

def add_salt_pepper_noise(image):
    row,col,ch = image.shape
    s_vs_p = 0.8
    amount = 0.7
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[coords] = 0
    return out

print("hi")


# In[38]:

TRAIN_DIR = 'C:/Users/PagolPoka/Desktop/Deep_Learning_Project/dataset/training_set'
   

def FetchImg():
    training_data_list = []
    label_list = []

      
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        if folder != '.DS_Store':
            Train_Path = TRAIN_DIR + '/' + folder
            #print(Train_Path)
            
            for img in tqdm(os.listdir(Train_Path)):
                
                label_list.append(img)
                path = os.path.join(Train_Path,img)
                img = cv2.imread(path)
                training_data_list.append(img)
                
    return training_data_list, label_list


# In[41]:

CleanPictures, Lables = FetchImg()
NoisyPictures = []
Lables_list = []

for img in CleanPictures:
    noiseImg = add_salt_pepper_noise(img)
    NoisyPictures.append(noiseImg)
    
for lbl in Lables:
    Lables_list.append(lbl)
    



# In[ ]:

path = "C:/Users/PagolPoka/Desktop/test_set/Noise/"
for i in range (len(NoisyPictures)):
    plt.figure()
    plt.imshow(NoisyPictures[i], cmap=plt.cm.binary)
    plt.show()
    cv2.imwrite(str(path)+ str(Lables_list[i])+'.png',NoisyPictures[i])
    





# In[ ]:



