"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
from statistics import mean, median
import matplotlib.pyplot as plt
import statistics
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
from skimage.transform import rescale
print(f"Size of Input image:\t\t{digits.images[0].shape}\t\n")
###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images


rescaled_images = np.asarray([rescale(img, 0.25, anti_aliasing=False) for img in digits.images])
    
    #print(rescaled_images)




data = rescaled_images.reshape((len(rescaled_images), -1))

#Lets print the shape of the reshaped image
print(f"Size of Rescaled image:   {rescaled_images[0].shape}\n")


# Create a classifier: a support vector classifier
param_grid = {'gamma': [1,0.5, 0.2, 0.1,0.07,0.05,0.03,0.01, 0.001, 0.0001], 'C': [0.1, 1, 20, 100,200,500,700,1000,1200,2000] } 
best_accuracy=[-1,-1,-1]
#most_accurate_model = None
for GAMMA in param_grid['gamma']:
    for C in param_grid['C']:
        hyper_params = {'gamma':GAMMA, 'C':C}
        print(f"hyperParameters:{hyper_params}")
        clf = svm.SVC()
        clf.set_params(**hyper_params)
# Split data into 40% train,30% test. 30% dev  subsets
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=0.5, random_state=42
)
    X_dev, X_test, y_dev, y_test = train_test_split(
    X_dev_test, y_dev_test, test_size=0.5, random_state=42
)

#fit data
    clf.fit(X_train, y_train)


 # Predict the value of the digit on the test subset

    accuracy_dev= metrics.accuracy_score(y_dev, clf.predict(X_dev))
   
    acc_test= metrics.accuracy_score(y_test,clf.predict(X_test))
   
    acc_train= metrics.accuracy_score(y_train,clf.predict(X_train))

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.
############################
    print(f"{clf} Train Accuracy: {acc_train*100:.2f} Dev Accuracy {accuracy_dev*100:.2f} Test Accuracy: {acc_test*100:.2f}\n")
    mean_accuracy= (accuracy_dev+acc_test+acc_train)/3
    max_accuracy= max(accuracy_dev,acc_test, acc_train)
    median_accuracy =statistics.median([accuracy_dev,acc_test, acc_train])
    print(f"\n mean Accuracy :{mean_accuracy}\n")
    print(f"max_Accuracy: {max_accuracy}\n")
    print(f"median_Accuracy:{median_accuracy}")
   
    if accuracy_dev > best_accuracy[1]:
        best_accuracy = [acc_train,accuracy_dev,acc_test]
        most_accurate_model = hyper_params
print(f"Required Best Accuracy :\n {most_accurate_model}\n Training Accuracy:{best_accuracy[0]*100:.2f}; Dev Accuracy:{best_accuracy[1]*100:.2f}; Test Accuracy:{best_accuracy[2]*100:.2f};\n")

