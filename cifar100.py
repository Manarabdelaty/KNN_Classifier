# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:03:30 2018

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt

# File path for the cifar-100 dataset
file_path = "cifar-100-python/"

# Number of classfication classes
number_classes = 20

# Reads cifar dataset files and returns a dictionary
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# loads classes from dataset file and returns a list of labels for the 20 superclasses
def load_classes(file_name):
    # reads the file
    dict = unpickle(file_name)
    # superclass labels 
    raw = dict[b'coarse_label_names']
    # convert from binary string to normal string
    names = [x.decode('utf-8') for x in raw]  
    return names

# Prints classes labels
def print_classes(labels):
    for i in range(len(labels)):
        print(i ," " , labels[i])

# Converts pixel value to a value between 0 ... 1
def convert_images(raw):
    raw_float = np.array(raw,dtype = float) /255.0
    #images = raw_float.reshape([-1,3, 32, 32])
    #print(images[0])
    #images = images.transpose([0, 2, 3, 1])
    return raw_float

# loads trainig images and their labels
def load_data(file_name):
    # reads file
    data = unpickle(file_name)
    # raw images. (N x 3072) matrix
    raw_imgs = data[b'data']     
    # class label of each image.  (N x 1) matrix                  
    labels  = np.array(data[b'coarse_labels'])
    # Converts pixel values to a value between 0...1     
    images = convert_images(raw_imgs)
    return images, labels

# slices trainining data and their labels. Returns a subset with the specified size 
def slice_data(data, labels, size):
    # Subset of the data matrix   
    train_set    = []
    # The corresponding labels of the train_set
    train_labels = np.zeros((number_classes,1), dtype = np.int)
    # num. of images taken from each class so they are all equally represented 
    imgs_per_class = size//number_classes
    
    # loop over all classes
    for i in range(number_classes):
        # to iterate over data matrix
        indx = 0
        images = np.zeros((imgs_per_class, 3072))
        # extract number of images from class i
        for k in range(imgs_per_class): 
            # searches for an image from the class
            while (labels[indx] != i):
               indx = indx + 1 
            images[k] = data[indx]
            indx = indx +1
        # add image to train_set 
        train_set.append(images)
        # add image label to labels set
        train_labels[i] = i
            
    return train_set, train_labels

# Slices data to n-folds. Returns train_set and valid_set at round r 
def crossfold(data, labels, n, r):
    imgs_per_class = len(data[0]) // n
    
    Xval = np.zeros(((imgs_per_class*number_classes), 3072))
    Yval = np.zeros((imgs_per_class * number_classes))
    Xtr  = np.zeros(((imgs_per_class*number_classes*2) , 3072))
    Ytr  = np.zeros((imgs_per_class *number_classes*2))

    for i in range(number_classes): 
        Xval[i*imgs_per_class:i*imgs_per_class+imgs_per_class, :] = data[i][r*imgs_per_class:r*imgs_per_class+imgs_per_class, :]
        Yval[i*imgs_per_class:i*imgs_per_class+imgs_per_class] = i
   
        arr = np.concatenate((data[i][:r*imgs_per_class, :],
                              data[i][r*imgs_per_class+imgs_per_class:, :]))
        
        Xtr [i*(2*imgs_per_class):(i*imgs_per_class+imgs_per_class)*2, :] = arr
        Ytr [i*(2*imgs_per_class):(i*imgs_per_class+imgs_per_class)*2]    = i
   
    return Xtr, Ytr, Xval, Yval;

def ccrn(Ypredicted, Ytest):
    Ytest_l = Ytest.tolist()
    print(Ytest)
    print(Ypredicted)
    class_occurence  = [ Ytest_l.count(i) for i in range(number_classes)]  
    correct_y = np.zeros((number_classes), dtype = np.float)
    for i in range(number_classes): 
        for j in range(len(Ytest)):
            if ( Ytest[j] == Ypredicted[j] and Ytest[j] == i):
                correct_y[i] = correct_y[i] + 1
    ccrn = correct_y / class_occurence
    print("class occurance", class_occurence)
    print("correct_y", correct_y)
    return ccrn
    
                
    
    
def pltacc(acc, K,fig_num = 0):
    plt.figure(fig_num)
    plt.title("Cross Validation on K")
    plt.xlabel("K")
    plt.ylabel("cross fold accuracy")
    plt.plot(acc, K , '-o')
    plt.show()
    plt.savefig('cross_valid.png')
    
    