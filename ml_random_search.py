import numpy as np
import sys

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from src.dataLoading import dataLoader
from src.featureExtraction import runPCA, mix_gauss, K_means, handcraftedFeaturesExtractor
from sklearn.model_selection import GridSearchCV


def svm_params():
    # Create an SVM
    svm = SVC()

    # Set up gridsearch params
    param_grid = {
        'C': np.logspace(-3, 3, num=7),
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, num=7)),
        'degree': np.arange(2, 6),
        'probability': [True, False],
        'class_weight': [None, 'balanced']
    }

    return svm, param_grid

def gboost_params():
    # define the parameter grid for the grid search
    param_grid = {
        'learning_rate': np.linspace(0.01, 1, num=20),
        'n_estimators': np.arange(50, 500, 50),
        'max_depth': np.arange(1, 11),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 6),
        'subsample': np.linspace(0.5, 1, num=6),
        'max_features': [None, 'sqrt', 'log2']
    }

    # perform the grid search
    gb = GradientBoostingClassifier()

    return gb, param_grid


def perform_randomsearchCV(model, param_grid, train_data, test_data, train_labels, test_labels, n_iter=100):

    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(train_data, train_labels)

    # print the best parameters and score
    print("Best parameters: ", random_search.best_params_)
    print("Best train score: ", random_search.best_score_)
    print("Best test score: ", random_search.score(test_data, test_labels))
    

def load_data(dtype="vector"):
    vectors, images, labels = dataLoader()

    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, train_size=0.8, stratify=labels)

    if dtype="vector":
        X_train, X_test, y_train, y_test = train_test_split(vectors, labels, train_size=0.8, stratify=labels)
    elif dtype="images":
        X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=0.8, stratify=labels)


if __name__ == "__main__":


    if sys.argv[1] == "pca":
        X_train, X_test, y_train, y_test = load_data("vector")
        x_train, x_test = runPCA(X_train, X_test)
    elif sys.argv[1] == "gau":
        X_train, X_test, y_train, y_test = load_data("images")
        x_train, x_test = mix_gauss(X_train, X_test)
    elif sys.argv[1] == "kmeans":
        X_train, X_test, y_train, y_test = load_data("images")
        x_train, x_test = K_means(X_train, X_test)
    elif sys.argv[1] == "hand":
        X_train, X_test, y_train, y_test = load_data("images")
        x_train, x_test = handcraftedFeaturesExtractor(X_train, X_test)

    print("data loaded")

    if sys.argv[2] == "svm":
        svm, param_grid = svm_params()
        perform_randomsearchCV(svm, param_grid, x_train, x_test, y_train, y_test)
    elif sys.argv[2] == "gb":
        gb, param_grid = gboost_params()
        perform_randomsearchCV(gb, param_grid, x_train, x_test, y_train, y_test)
    elif sys.argv[2] == "test":
        svm, param_grid = svm_params()
        perform_randomsearchCV(svm, param_grid, x_train, x_test, y_train, y_test, n_iter=1)
