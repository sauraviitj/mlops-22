import argparse
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from utils import data_preprocess
from utils import train_dev_test_split
from utils import h_param_tuning
from joblib import dump

parser = argparse.ArgumentParser()
parser.add_argument('--clf_name', type=str)
parser.add_argument('--random_state', type=int)
args = parser.parse_args()



digits = datasets.load_digits()

data, label = data_preprocess(digits)

del digits

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.6, 1, 2, 5, 7, 10]
h_param_comb_svm = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

min_samples_split_list = [1,4,6,12]
min_samples_leaf_list = [1,3,5,10]

h_param_comb_dtree = [{"min_samples_leaf": g, "min_samples_split": c}for g in min_samples_leaf_list for c in min_samples_split_list]

model_of_choices = [svm.SVC(),tree.DecisionTreeClassifier()]
hyper_param = [h_param_comb_svm,h_param_comb_dtree]
metric=metrics.accuracy_score

x = list(set(label))



if args.clf_name in "svm":
    clf = model_of_choices[0]
elif args.clf_name in "tree":
    clf = model_of_choices[1]
X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
    data, label, 0.6,.2,args.random_state
)



best_model, best_metric, best_h_params = h_param_tuning(hyper_param[0], clf, X_train, y_train, X_dev, y_dev, X_test, y_test, metric)
prediction = best_model.predict(X_test)

model_loc = f'{best_model}.joblib'
dump(best_model, model_loc)
accuracy = metrics.accuracy_score(y_test, prediction)
macrof1 = metrics.f1_score(y_test, prediction, average='macro')

out_text = [f'test accuracy: {accuracy}', f'test macro-f1: {macrof1}',
            f'model saved at ./{model_loc}']

filename = f"{args.clf_name}_{args.random_state}.txt"

with open(filename, 'w') as file:
    file.writelines("% s\n" % data for data in out_text)