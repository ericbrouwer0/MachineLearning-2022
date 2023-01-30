import cv2
import numpy as np
import pandas as pd
import pickle
import os

def MNISTreader(samplesPerClass = 1000, random_state=None):
    df1=pd.read_csv('Data/mnist_train.csv')
    df2=pd.read_csv('Data/mnist_test.csv')
    df = pd.concat([df1, df2])

    df = df.groupby('label').sample(n=samplesPerClass, random_state=random_state)
    labels = df['label'].to_list()
    df.drop(['label'], axis=1, inplace=True)

    vectors = df.to_numpy()
    images = []

    for vec in vectors:
        img = np.reshape(vec, (28,28))
        images.append(img)
    
    return vectors, images, labels



def chineseMNISTreader():
    df=pd.read_csv('Data/chineseMNIST.csv')

    # get rid of the non 0-9 classes here
    df = df[df.label < 10]
    len(df)

    df = df.sort_values(by=['label'])
    labels = df['label'].to_list()
    characters = df['character']
    df.drop(['label'], axis=1, inplace=True)
    df.drop(['character'], axis=1, inplace=True)

    # "images" is the np array of images
    vectors = df.to_numpy()

    return vectors, labels


# function to find a bounding box for the character on the image, crop it 
# down to that bounding box, and resize to 28x28
def resizeAndCrop(dataset):

    imageDataset = [] #
    vectorDataset = np.zeros((len(dataset), 784))

    for idx, vector in enumerate(dataset):
        img = np.reshape(vector, (64,64)).astype('uint8')

        # threshold 
        ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8) #also needs to be tuned
        img_dil = cv2.dilate(img_bin, kernel, iterations=1)
        #img_smth = gaussian_filter(img_bin, sigma= sigma)
        #img_smth = cv2.GaussianBlur(img_dil,(3,3),sigmaX=0.6)
        img_smth = cv2.blur(img_dil,(2,2))

        # get bounds of white pixels
        white = np.where(img_bin==255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        yrange = ymax-ymin
        xrange = xmax-xmin

        # take the max of the width or height of the image, we'll use this to define the cropping
        Range = yrange if yrange>xrange else xrange
        if Range%2 != 0: # make sure its not odd
            Range += 1

        x_center = int(np.round((xmax+xmin)/2))
        y_center = int(np.round((ymax+ymin)/2))

        size = int(Range/2 + 5) # pad all sides by 5 pixels

        # the dimensions of the bounding box (prevent going off the edges of the image)
        top = y_center-size if y_center-size > 0 else 0
        bottom = y_center+size if y_center+size < 64 else 64
        left = x_center-size if x_center-size > 0 else 0
        right = x_center+size if x_center+size < 64 else 64

        # crop the image and resize it
        crop = img_smth[top : bottom, left : right]
        resized_crop = cv2.resize(crop, (28,28))

        # add to datasets
        imageDataset.append(resized_crop)
        new_vector = np.reshape(resized_crop, (784,))
        vectorDataset[idx] = new_vector

    return vectorDataset, imageDataset


def loadNormalMNIST(samplesPerClass = 1000, random_state=None):
    mnistVectors, mnistImages, mnistLabels = MNISTreader(samplesPerClass = 1000, random_state=random_state)    
    
    return mnistVectors, mnistImages, mnistLabels


def loadChineseMNIST():
    vectors, chineseLabels = chineseMNISTreader()
    chineseVectors, chineseImages = resizeAndCrop(vectors)
    #processedImages = processedImages

    return chineseVectors, chineseImages, chineseLabels


# seperate the integer labels of both datasets by making them into strings
def makeNewLabels(oldLabels, chinese=False):
    newLabels = []
    for lab in oldLabels:
        if chinese == True:
            newLab = "chinese_"+str(lab)
        else:
            newLab = "mnist_"+str(lab)
        newLabels.append(newLab)
    return newLabels

# load both of the datasets in one set
# returns the images as vectors, arrays and their (string) labels
# first half of data is the classic mnist digits, second half is the chinese mnist digits
def dataLoader(refresh_data = False, mnist_only=False, chinese_mnist_only=False):
    data_dir = 'Data/processed_data.pkl'
    if(mnist_only == True and chinese_mnist_only == False):
        data_dir = 'Data/processed_data_mnist.pkl'
    elif(mnist_only == False and chinese_mnist_only == True):
        data_dir = 'Data/processed_data_chinese.pkl'
        
    if(not refresh_data and os.path.exists(data_dir)):
        with open(data_dir, 'rb') as f:
            allVectors, allImages, allLabels = pickle.load(f)
        return allVectors, allImages, allLabels
    
    mnistVectors, mnistImages, mnistLabels = loadNormalMNIST()
    chineseVectors, chineseImages, chineseLabels = loadChineseMNIST()

    newChineseLabels = makeNewLabels(chineseLabels, chinese=True)
    newMnistLabels = makeNewLabels(mnistLabels, chinese=False)

    allVectors, allImages, allLabels = [], [], []
    if(mnist_only == False and chinese_mnist_only == False):
        allVectors = np.concatenate([mnistVectors, chineseVectors])
        allImages = np.concatenate([mnistImages, chineseImages])
        allLabels = np.concatenate([newMnistLabels, newChineseLabels])
    elif(mnist_only):
        allVectors = mnistVectors
        allImages = mnistImages
        allLabels = newMnistLabels
    else:
        allVectors = chineseVectors
        allImages = chineseImages
        allLabels = newChineseLabels
    
    with open(data_dir, 'wb') as f:
        pickle.dump([allVectors, allImages, allLabels], f)

    return allVectors, allImages, allLabels
