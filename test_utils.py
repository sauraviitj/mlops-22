import sys, os
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm

sys.path.append(".")

from utils import get_all_h_param_comb, tune_and_save
from sklearn import svm, metrics
digits = datasets.load_digits()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clf_name', type=str)
parser.add_argument('--random_state', type=int)
args = parser.parse_args()

seed = int(args.random_state)
model = args.clf_name
print(model)

# test case to check if all the combinations of the hyper parameters are indeed getting created
'''
def test_get_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)
'''
'''
def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)
    return h_param_comb

def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train
def train_test_split_test(random_st,test_sz):
    
    x = data.images.reshape((len(data.images), -1))
    return x,data.target 
    x_train, x_test, y_train, x_test = train_test_split(
        data, label, random_state = random_st, test_size=test_sz,shuffle=True
    )
    
    return x_train, x_test,y_train,x_test
'''
def data_preprocess(data):
    # flatten the images
   
    x = data.images.reshape((len(data.images), -1))
    return x,data.target

data, label = data_preprocess(digits)
def def_train_test_split(random_st,test_sz):
    
    
    x_train, x_test, y_train, x_test = train_test_split(
        data, label, random_state = random_st, test_size=test_sz,shuffle=True
    )
    
    return x_train, x_test,y_train,x_test
def test_equal():
    random_state_1 = 50
    random_state_2 = 50
    test_size = 30
    x_train1, x_test1, y_train1, y_test1 = def_train_test_split(test_size,random_state_1)
    x_train2, x_test2, y_train2, y_test2 = def_train_test_split(test_size,random_state_2)
    assert  np.array_equal(x_train1,x_train2)
    assert  np.array_equal(y_train1 ,y_train2)
    assert  np.array_equal(x_test1 , x_test2)
    assert  np.array_equal(y_test1 ,y_test2)
    
    
    
def test_unequal():
    random_state_1 = 40
    random_state_2 = 30
    test_size = 30
    x_train1, x_test1, y_train1, y_test1 = def_train_test_split(test_size,random_state_1)
    x_train2, x_test2, y_train2, y_test2 = def_train_test_split(test_size,random_state_2)
    assert not np.array_equal(x_train1, x_train2)
    assert not np.array_equal(y_train1, y_train2)
    assert not np.array_equal(x_test1, x_test2)
    assert not np.array_equal(y_test1, y_test2)
'''
def test_tune_and_save():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)

    assert actual_model_path == model_path
    assert os.path.exists(actual_model_path)
    assert type(load(actual_model_path)) == type(clf)


def test_not_biased():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert len(set(predicted))!=1


def test_predicts_all():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert set(predicted) == set(y_test)
'''

