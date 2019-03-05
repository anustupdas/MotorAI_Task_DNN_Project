
# coding: utf-8

# In[3]:

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from sklearn.utils import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm     # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
from sklearn.cross_validation import train_test_split #Splitting the data into Train and Test
from PIL import Image      # Converting the ImageNumpyArrayy into Image object

import matplotlib.pyplot as plt #Ploting pictures and Chart

from keras import backend as K
K.set_image_dim_ordering('th')

from keras import callbacks
from keras.utils import np_utils
from keras.models import Model 
from keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D

TRAIN_DIR = 'C:/Users/PagolPoka/Desktop/Deep_Learning_Project/dataset/training_set'
NOISY_TRAIN_DIR = 'C:/Users/PagolPoka/Desktop/test_set/Noise'
TEST_DIR = 'C:/Users/PagolPoka/Desktop/Deep_Learning_Project/dataset/test_set'
IMG_SIZE = 128
LR = 1e-3
num_channel=3
num_epoch=100


# In[4]:

#Fetches all the training pictures from the directory
def create_train_data():
    training_data_list = []
    labels_list = []
  
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        if folder != '.DS_Store':
            Train_Path = TRAIN_DIR + '/' + folder
            #print(Train_Path)
            
            for img in tqdm(os.listdir(Train_Path)):
                labels_list.append(img) 
                path = os.path.join(Train_Path,img)
                img = cv2.imread(path,cv2.IMREAD_COLOR) #Reading the Image in color
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) #Resizing the Image 
                training_data_list.append(img)
                
    #Preprocesing the Data for fitting it into the DL Model
    training_data_raw = training_data_list
    training_data = np.array(training_data_list)    
    training_data = training_data.astype('float32')  
    training_data /= 255
    print ('Shape: ',training_data.shape)
    
    return training_data, labels_list, training_data_raw

#Fetches all the Noisy training pictures from the directory [Same as Above]
def create_noisy_train_data():
    noisy_training_data_list = []
    noise_labels_list = []
  
    for folder in tqdm(os.listdir(NOISY_TRAIN_DIR)):
        if folder != '.DS_Store':
            Train_Path = NOISY_TRAIN_DIR + '/' + folder
            #print(Train_Path)
            
            for img in tqdm(os.listdir(Train_Path)):
                noise_labels_list.append(img) 
                path = os.path.join(Train_Path,img)
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                noisy_training_data_list.append(img)
                
    

    training_data_raw = noisy_training_data_list
    noisy_training_data = np.array(noisy_training_data_list)    
    noisy_training_data = noisy_training_data.astype('float32')  
    noisy_training_data /= 255
    print ('Shape noise: ',noisy_training_data.shape)
    
    return noisy_training_data, noise_labels_list, training_data_raw

#Fetches all the Noisy test pictures to be converted into clean from the directory [Same as Above]
def create_noisy_test_data():
    noisy_test_data_list = []
    noise_test_labels_list = []
  
    for folder in tqdm(os.listdir(TEST_DIR)):
        if folder != '.DS_Store':
            Train_Path = TEST_DIR + '/' + folder
            #print(Train_Path)
            
            for img in tqdm(os.listdir(Train_Path)):
                noise_test_labels_list.append(img) 
                path = os.path.join(Train_Path,img)
                img = cv2.imread(path,cv2.cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                noisy_test_data_list.append(img)
                
    
    training_data_raw = noisy_test_data_list
    noisy_test_data = np.array(noisy_test_data_list)    
    noisy_test_data = noisy_test_data.astype('float32')  
    noisy_test_data /= 255
    print ('Shape noise: ',noisy_test_data.shape)
    
    return noisy_test_data, noise_test_labels_list, training_data_raw
    


# In[5]:

#Creating the clean training data and splitting it into train and test
Train_Data, Lables, Train_imgs = create_train_data()
print ("Shape of X_train_clean before reshaping: ",Train_Data.shape)

#Reshaping the data to the input format for the DL model based on the number of channels.
#num_channel==1 [BLACK AND WHITE]
#num_channel==3 [COLOR]
if num_channel==1:
    if K.image_dim_ordering()=='th':
        Train_Data= np.expand_dims(Train_Data, axis=1)
        #print (Train_Data.shape)
    else:
        Train_Data= np.expand_dims(Train_Data, axis=4) 
        #print (Train_Data.shape)       
else:
    if K.image_dim_ordering()=='th':
        Train_Data=np.rollaxis(Train_Data,3,1)
        #print (Train_Data.shape)
        

X_train, X_test, y_train, y_test = train_test_split(Train_Data, Lables, test_size=0.2, random_state=2)

print("Shape of X_train_Clean after reshaping and splitting: ",X_train.shape)
print("Shape of X_test_Clean after reshaping and splitting: ",X_test.shape)

# Defining the Input_Shape for the model
input_shape=Train_Data[0].shape


# In[6]:

#Visualising some of the clean training clean Images
print("Total number of images: ",len(Train_imgs))
print("Matrix representation of 67th image: ",Train_imgs[67])


#Displaying 149th to 155th image from the training clean data
for i in range (149,155):
    plt.figure()
    plt.imshow(Train_imgs[i], cmap=plt.cm.binary)
    lable = Lables[i]
    plt.xlabel(lable)
        
    plt.show()    
    print("The train label is: ",Lables[i])


# In[22]:

#Creating the noisy training data and splitting it into train and test
Noisy_Train_Data, Noisy_Lables, Noisy_Train_imgs = create_noisy_train_data()
print ("Shape of X_train_Noisy before reshaping: ",Noisy_Train_Data.shape)

#Reshaping the Data to fit into the Autoencoder model.
if num_channel==1:
    if K.image_dim_ordering()=='th':
        Noisy_Train_Data= np.expand_dims(Noisy_Train_Data, axis=1)
        #print (Noisy_Train_Data.shape)
    else:
        Noisy_Train_Data= np.expand_dims(Noisy_Train_Data, axis=4) 
        #print (Noisy_Train_Data.shape)       
else:
    if K.image_dim_ordering()=='th':
        Noisy_Train_Data=np.rollaxis(Noisy_Train_Data,3,1)
        #print (Noisy_Train_Data.shape)
        

X_train_Noisy, X_test_Noisy, y_train_Noisy, y_test_Noisy = train_test_split(Noisy_Train_Data, Noisy_Lables, test_size=0.2, random_state=2)

print("Shape of X_train_Noisy after reshaping and splitting: ",X_train_Noisy.shape)
print("Shape of X_test_Noisy after reshaping and splitting: ",X_test_Noisy.shape)

# Defining the model
input_shape=Train_Data[0].shape
print("Input shape of the encoder model: ", input_shape)


# In[12]:

#Visualising some of the noisy training Images
print("Total number of images: ",len(Noisy_Train_imgs))
print("Matrix representation of 67th image: ",Noisy_Train_imgs[67])

#Displaying 149th to 155th image from the training noisy data
for i in range (149,155):
    plt.figure()
    plt.imshow(Noisy_Train_imgs[i], cmap=plt.cm.binary)
    lable = Noisy_Lables[i]
    plt.xlabel(lable)
        
    plt.show()    


# In[17]:

#Creating the noisy data that is to be cleaned 
Noisy_Test_Data, Noisy_Test_Lables, Noisy_Test_imgs = create_noisy_test_data()
print ("Shape of X_val_Noisy before reshaping",Noisy_Test_Data.shape)

#Reshaping the Data to fit into the Autoencoder model.
if num_channel==1:
    if K.image_dim_ordering()=='th':
        Noisy_Test_Data= np.expand_dims(Noisy_Test_Data, axis=1)
        #print (Noisy_Test_Data.shape)
    else:
        Noisy_Test_Data= np.expand_dims(Noisy_Test_Data, axis=4) 
        #print (Noisy_Test_Data.shape)       
else:
    if K.image_dim_ordering()=='th':
        Noisy_Test_Data=np.rollaxis(Noisy_Test_Data,3,1)
        #print (Noisy_Test_Data.shape)
        
Train_Data, Lables
X_val_Noisy, y_val_Noisy = (Noisy_Test_Data, Noisy_Test_Lables) 

print("Shape of X_val_Noisy after reshaping",X_val_Noisy.shape)
print("Number os lables: ", len(y_val_Noisy))



# In[25]:

#Visualising some of the noisy input Images to be cleaned
print(len(Noisy_Test_imgs))
print(Noisy_Test_imgs[67])

for i in range (72,76):
    plt.figure()
    plt.imshow(Noisy_Test_imgs[i], cmap=plt.cm.binary)
    lable = Noisy_Test_Lables[i]
    plt.xlabel(lable)
        
    plt.show()    


# In[27]:

#The Model Definaion

input_img = Input(input_shape)  # adapt this if using `channels_first` image data format
x = Conv2D(10, (5, 5), activation='tanh', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(20, (2, 2), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = x

x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(20, (2, 2), activation='tanh', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(10, (5, 5), activation='tanh', padding='same')(x)
x = Conv2D(10, (5, 5), activation='tanh', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])


# In[88]:

#####################Training the model with noisy test image###########################
print("Training")
filename='DenoiseImg_Sigmoid_model_train.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='min')

filepath="DenoiseImg-Sigmoid-Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]


# In[89]:

#Training the Model
hist = autoencoder.fit(X_train_Noisy, X_train,
                epochs=num_epoch,
                verbose=1,
                batch_size=25,
                shuffle=True,
                validation_data=(X_test_Noisy, X_test),
                callbacks=callbacks_list)


# In[90]:

#Sample Prediction with the input noisy image to be cleaned
testSample = X_val_Noisy[60]
print(testSample.shape)
test_predict = autoencoder.predict(np.array([testSample]))[0]


# In[91]:

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)

plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train_loss','val_loss'], loc=10, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train_acc','val_acc'],loc=10)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.show()


# In[92]:

#Converting the noisy input and clean output into their old form to be displayed
print(testSample.shape)
testSample = np.array((testSample * 255 )[0], dtype=np.uint8)
img = Image.fromarray(testSample)

test_predict = np.array((test_predict * 255 )[0], dtype=np.uint8)
Noise_img = Image.fromarray(test_predict)



# In[93]:

#Comparing the noisy input with the clean Output
plt.figure()
plt.imshow(img, cmap=plt.cm.binary)        
plt.show()    


plt.figure()
plt.imshow(Noise_img, cmap=plt.cm.binary)        
plt.show()


# In[84]:

#cleaning all the noisy images using the trained model and storing them in a list along with their respective labels
CleanPictures = []
CleanPictureArray = []
Lables_list = []

for i in range(len(X_val_Noisy)):
    Val_Sample = X_val_Noisy[i]
    Val_predict = autoencoder.predict(np.array([Val_Sample]))[0]
    
    Val_predict = np.array((Val_predict * 255 )[0], dtype=np.uint8)
    Clean_img = Image.fromarray(Val_predict)
    CleanPictures.append(Clean_img)
    CleanPictureArray.append(Val_predict)

for lbl in y_val_Noisy:
    Lables_list.append(lbl)
    


# In[28]:

#print("Number of images cleaned: "len(CleanPictures))
#print(len(CleanPictureArray))
#print(len(Lables_list))


# In[86]:

#Writting the output to a file
path = "C:/Users/Sumit Kundu/Image Classification using CNN/dataset/Cleaned_MIX/"
for i in range (len(CleanPictures)):
    plt.figure()
    plt.imshow(CleanPictures[i], cmap=plt.cm.binary)
    plt.show()
    cv2.imwrite(str(path)+ str(Lables_list[i]),CleanPictureArray[i])


# In[ ]:



