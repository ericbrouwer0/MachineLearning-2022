from sklearn.decomposition import PCA


# takes in the test and train data, returns PCA feature-extracted versions
def runPCA(X_train, X_test, n_components = 85):
    sklearn_pca = PCA(n_components=n_components)   # we find 85 components is best

    pca_X_train = sklearn_pca.fit_transform(X_train)  # fit + transform
    pca_X_test = sklearn_pca.transform(X_test)        # transform

    return pca_X_train, pca_X_test




################################################################
##############   HANDCRAFTED FEATURE EXTRACTION   ##############
################################################################






