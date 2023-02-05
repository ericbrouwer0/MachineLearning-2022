import tensorflow as tf
from tensorflow.keras import datasets, layers, regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
from src.dataLoading import dataLoader
from sklearn.model_selection import train_test_split
import itertools

print("GPU Available: ", tf.test.is_gpu_available())

vectors, images, labels = dataLoader(mnist_only=False, chinese_mnist_only=False)  # 784-long vectors, 28*28 images and mnist/chinese labels

encoder = LabelBinarizer()
transfomed_labels = encoder.fit(np.unique(labels))
# split the vectors (for PCA, CNN would use images)
# "stratify" makes sure theres a balance of each class in the test/train sets
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, train_size=0.8, stratify=labels)
y_train_t = encoder.transform(y_train)
y_test_t = encoder.transform(y_test)
X_train = np.reshape(X_train, (X_train.shape[0],28,28,1))
X_test = np.reshape(X_test, (X_test.shape[0],28,28,1))


EPOCHS = 100
BATCH_SIZE = [32, 64]
l1 = [0.1, 0.01, 0.001]
l2 = [0.1, 0.01, 0.001]
dropouts = [0.5, 0.2]
conv_models = [
    [32, 64],
    [32, 64, 128],
    [16, 32],
    [16, 32, 64],
    [64, 128],
    # [64, 128, 256]
]
add_batch_norm = [True, False]
dense_models = [
    [32],
    # [128, 64],
    [64],
    [64, 32],
    # [128, 32],
    [128]
]


parameters = list(itertools.product(*[conv_models, add_batch_norm, dense_models, dropouts, BATCH_SIZE]))
print(f'total number of combinations: {len(parameters)}')


def create_model_2(input_shape, output_shape, conv_layers, batch_norm, dense_layers, dropout, steps_per_epoch):
    model = tf.keras.models.Sequential()
    for idx, val in enumerate(conv_layers):
        if(idx == 0):
            model.add(layers.Conv2D(val, (3, 3), activation=None, input_shape=input_shape))
        else:
            model.add(layers.Conv2D(val, (3, 3), activation=None))
        if(batch_norm):
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout)),
    for idx, val in enumerate(dense_layers):
        model.add(layers.Dense(val, kernel_regularizer=regularizers.L1L2(l1=0.00, l2=0.00), activation='relu'))
        model.add(layers.Dropout(dropout)),
    model.add(layers.Dense(output_shape))
       
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.001,
      decay_steps=steps_per_epoch*1000,
      decay_rate=1,
      staircase=False)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model



def cross_validate(conv_model, batch_norm, dense_model, dropout, batch_size, k_folds=10):
    acc_per_fold = []
    loss_per_fold = []
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_n = 1
    for train, test in kfold.split(X_train, y_train_t):
        print(f'FOLD {fold_n}')
        model = create_model_2(X_train.shape[1:], y_train_t.shape[-1], conv_model, batch_norm, dense_model, dropout,  X_train.shape[0]//batch_size)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        ]
        history = model.fit(
            X_train[train],
            y_train_t[train],
            epochs=EPOCHS,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=2,
            callbacks=callbacks
        )
        test_loss, test_acc = model.evaluate(X_train[test], y_train_t[test], verbose=2)
        print(f'Score for fold {fold_n}: {model.metrics_names[0]} of {test_loss}; {model.metrics_names[1]} of {test_acc*100}%')
        acc_per_fold.append(test_acc * 100)
        loss_per_fold.append(test_loss)
        fold_n += 1
    return [np.mean(acc_per_fold), np.mean(loss_per_fold)]


means = [cross_validate(conv_model, batch_norm, dense_model, dropout, batch_size) for conv_model, batch_norm, dense_model, dropout, batch_size in parameters]

print(means)
print(parameters)

top_params = [parameters[idx] for idx in np.argsort(np.array(means)[:,0])[::-1][:3]]
print('Top model Configurations')
print(top_params)

top_means = [means[idx] for idx in np.argsort(np.array(means)[:,0])[::-1][:3]]
print('Top mean accuracies with their losses')
print(top_means)
