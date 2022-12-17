import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def MNISTreader(samplesPerClass = 1000, random_state=None):
    df1=pd.read_csv('Data/mnist_train.csv')
    df2=pd.read_csv('Data/mnist_test.csv')
    df = pd.concat([df1, df2])
    #
    df = df.groupby('label').sample(n=samplesPerClass, random_state=random_state)
    labels = df['label'].to_list()
    df.drop(['label'], axis=1, inplace=True)
    images = df.to_numpy()

    return images, labels



def chineseMNISTreader():

    df=pd.read_csv('Data/chineseMNIST.csv')

    # get rid of the non 0-9 classes here
    df = df[df.label < 10]
    len(df)

    labels = df['label'].to_list()
    characters = df['character']
    df.drop(['label'], axis=1, inplace=True)
    df.drop(['character'], axis=1, inplace=True)

    # "images" is the np array of images
    images = df.to_numpy()

    return images, labels



# function to find a bounding box for the character on the image, crop it 
# down to that bounding box, and resize to 28x28
def resizeAndCrop(dataset):

    newDataset = np.zeros((len(dataset), 784))

    for idx, vector in enumerate(dataset):
        img = np.reshape(vector, (64,64)).astype('uint8')

        # threshold 
        ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        """Sara's stuff"""
        kernel = np.ones((2, 2), np.uint8) #also needs to be tuned
        img_dil = cv2.dilate(img_bin, kernel, iterations=1)
        #img_smth = gaussian_filter(img_bin, sigma= sigma)
        #img_smth = cv2.GaussianBlur(img_dil,(3,3),sigmaX=0.6)
        img_smth = cv2.blur(img_dil,(2,2))

        """the rest is the same"""
        # get bounds of white pixels
        white = np.where(img_bin==255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        yrange = ymax-ymin
        xrange = xmax-xmin

        # take the max of the width or height of the image, we'll use this to define the cropping
        range = yrange if yrange>xrange else xrange
        if range%2 != 0: # make sure its not odd
            range += 1

        x_center = int(np.round((xmax+xmin)/2))
        y_center = int(np.round((ymax+ymin)/2))

        size = int(range/2 + 5) # pad all sides by 5 pixels

        # the dimensions of the bounding box (prevent going off the edges of the image)
        top = y_center-size if y_center-size > 0 else 0
        bottom = y_center+size if y_center+size < 64 else 64
        left = x_center-size if x_center-size > 0 else 0
        right = x_center+size if x_center+size < 64 else 64

        # crop the image and resize it
        crop = img_smth[top : bottom, left : right]
        resized_crop = cv2.resize(crop, (28,28))

        #new_vector = np.reshape(resized_crop, (784,))
        newDataset[idx] = resized_crop

    return newDataset

    return newDataset



def loadNormalMNIST(samplesPerClass = 1000, random_state=None):
    images, labels = MNISTreader(samplesPerClass = 1000, random_state=random_state)    
    
    return images, labels


def loadChineseMNIST():
    chineseImages, chineseLabels = chineseMNISTreader()
    processedImages = resizeAndCrop(chineseImages)

    processedImages = processedImages

    return processedImages, chineseLabels