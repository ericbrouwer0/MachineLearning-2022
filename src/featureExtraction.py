from sklearn.decomposition import PCA
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import cv2


################################################################
##########################     PCA    ##########################
################################################################

# takes in the test and train data, returns PCA feature-extracted versions
def runPCA(X_train, X_test, n_components = 85):
    sklearn_pca = PCA(n_components=n_components)   # we find 85 components is best

    pca_X_train = sklearn_pca.fit_transform(X_train)  # fit + transform
    pca_X_test = sklearn_pca.transform(X_test)        # transform

    return pca_X_train, pca_X_test



################################################################
##############   HANDCRAFTED FEATURE EXTRACTION   ##############
################################################################


### MIXTURE OF GAUSSIANS
def mix_gauss(x_train, x_test, n_components=10):
    GM = GaussianMixture(n_components = n_components)
    GM.fit(x_train) 
    # This is enough for the feature
    X_train_GM = GM.predict(x_train)
    X_test_GM = GM.predict(x_test)

    return X_train_GM, X_test_GM


### K-MEANS
def K_means(x_train, x_test, n_components=10):
    kmeans = KMeans(n_clusters=n_components)

    k = 1
    X_train_KNN = kmeans.fit_transform(x_train)
    X_test_KNN = kmeans.transform(x_test)
    #closest cluster will be another feature 
    closest_cluster = np.zeros(len(X_train_KNN[:,1]))
    for i in range(len(X_train_KNN[:,1])):
        min = np.min(X_train_KNN[i,:])
        index = np.where(X_train_KNN[i,:] == min)[0]
        closest_cluster[i] = int(index[0])
    X_train = (closest_cluster)

    closest_cluster = np.zeros(len(X_test_KNN[:,1]))
    for i in range(len(X_test_KNN[:,1])):
        min = np.min(X_test_KNN[i,:])
        index = np.where(X_test_KNN[i,:] == min)[0]
        closest_cluster[i] = int(index[0])
    X_test = closest_cluster

    return X_train, X_test


### LAPLACE EDGES
def edges_laplace(data_Images):
    edges = np.zeros(len(data_Images)) 
    for ii in range(len(data_Images)):
        img = data_Images[ii].astype('uint8')
        lapl = cv2.Laplacian(img, ddepth=-1) # depth -1: output same depth as source
        edges[ii] = np.sum(lapl)
    #plt.imshow(lapl, cmap='gray')
    return edges


### MEAN BRIGHTNESS
def mean_brightness(vectors):
    brightness = []
    for vec in vectors:
        brightness.append(np.sum(vec)/len(vec))
    return brightness


### NUMBER OF COUNTOURS (LENGTH OF COUNTOURS)
def n_Contours(data): 
    #Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity

    nu_lines = np.zeros(len(data)) 
    for ii in range(len(data)):
        img = data[ii].astype('uint8')
        img_bin = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        num_labels, labels = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        nu_lines[ii] = np.shape(num_labels)[0]

    return nu_lines


### NUMBER OF CIRCLES
def n_Circles(data):
    #IMAGE DATA
    circles = np.zeros(len(data))
    for ii in range(len(data)):
        img = data[ii].astype('uint8')
        img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        try:
            circles[ii] = len(cv2.HoughCircles(img_bin,cv2.HOUGH_GRADIENT_ALT, dp=1.5 , minDist = 7,
                            param1=10,param2=1)) 
        except:
            continue
    return circles


### HEIGHT/WIDTH RATIO
def heigh_width(data):
    #IMAGE DATA
    ratio = np.zeros(len(data))
    for ii in range(len(data)):
        img = data[ii].astype('uint8')
        img_bin = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        white = np.where(img_bin == 255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        ratio[ii] = (ymax-ymin)/(xmax-xmin)
    return ratio

### VALUES OF CENTER COLUMN / CENTER ROW
def center_col_row(data):
    #IMAGE DATA
    ratios = np.zeros(len(data))
    for ii in range(len(data)):
        img = data[ii].astype('uint8')
        img_bin = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] 
        white = np.where(img_bin != 0)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        yrange, xrange = ymax-ymin, xmax-xmin

        x_center = int(np.round((xmax+xmin)/2))
        y_center = int(np.round((ymax+ymin)/2))

        if xrange%2 != 0:
            x_sum = np.sum(img_bin[x_center, :]) + np.sum(img_bin[x_center-1, :])
            x_sum /= 2
        else: 
            x_sum = np.sum(img_bin[x_center,:])
        if yrange%2 != 0:
            y_sum = np.sum(img_bin[:,y_center]) + np.sum(img_bin[:, y_center-1])
            y_sum /= 2
        else: 
            y_sum = np.sum(img_bin[: ,y_center])
        
        if y_sum >= x_sum:
            ratios[ii] = 1
        else:
            ratios[ii] = 0
    return ratios


### COUNTING BLACK &&& WHITE ISLANDS
def islandCounter(data):
    length = len(data)
    whiteIslandsArray = np.zeros(length)
    blackIslandsArray = np.zeros(length)
    for ii in range(length):
        image = data[ii].astype('uint8')

        # get the external edges (white object on black background)
        ret, thresholdedIm = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _  = cv2.findContours(thresholdedIm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # get the number of white islands (just use the contours, this works fine)
        whiteIslands = np.zeros_like(thresholdedIm)
        numWhiteIslands = len(contours)
        whiteIslandsArray[ii] = numWhiteIslands

        # Draw contour onto blank mask
        mask = np.zeros_like(thresholdedIm)
        cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))
        # use XOR between the mask and the thresholded image to find the black islands
        blackIslands = cv2.bitwise_xor(thresholdedIm, mask)

        # get the number of black islands
        blackContours, _ = cv2.findContours(blackIslands, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        numBlackIslands = len(blackContours)
        blackIslandsArray[ii] = numBlackIslands

    return whiteIslandsArray, blackIslandsArray



####################################
# COMBINE ALL OF THE ABOVE:
def imgToVectors(images):
    vectors = []
    for img in images:
        vec = np.reshape(img, (784,))
        vectors.append(vec)
    return vectors


def handcraftedFeaturesExtractor(train_img, test_img):
    # turn the images into vectors, as some of the features are easier to make with vectors
    train_vec = imgToVectors(train_img)
    test_vec = imgToVectors(test_img)

    # I will append each feature to this, then transpose it at the end
    X_train = []
    X_test = []

    n_gaussians = 10
    MG_train, MG_test = mix_gauss(train_vec, test_vec, n_gaussians)
    X_train.append(MG_train)
    X_test.append(MG_test)

    n_clusters = 20
    KM_train, KM_test = K_means(train_vec, test_vec, n_clusters)
    X_train.append(KM_train)
    X_test.append(KM_test)

    X_train.append(edges_laplace(train_img))
    X_test.append(edges_laplace(test_img))

    X_train.append(mean_brightness(train_vec))
    X_test.append(mean_brightness(test_vec))

    X_train.append(n_Contours(train_img))
    X_test.append(n_Contours(test_img))

    X_train.append(n_Circles(train_img))
    X_test.append(n_Circles(test_img))

    X_train.append(heigh_width(train_img))
    X_test.append(heigh_width(test_img))

    X_train.append(center_col_row(train_img))
    X_test.append(center_col_row(test_img))
    
    trainWhites, trainBlacks = islandCounter(train_img)
    X_train.append(trainWhites)
    X_train.append(trainBlacks)
    testWhites, testBlacks = islandCounter(test_img)
    X_test.append(testWhites)
    X_test.append(testBlacks)    

    X_train = np.asarray(X_train).T
    X_test = np.asarray(X_test).T

    return X_train, X_test

