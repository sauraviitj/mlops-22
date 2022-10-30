import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from statistics import mean, median
import pandas as pd

def data_viz(data_to_viz):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, data_to_viz.images, data_to_viz.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


def data_preprocess(data):
    # flatten the images
    n_samples = len(data.images)
    x = data.images.reshape((n_samples, -1))
    return x, data.target


def train_dev_test_split(data, label, train_frac, dev_frac):
    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, x_test, y_test, metric):
    best_metric_train = -1.0
    best_model_train = None
    best_h_params_train = None
    best_metric_dev = -1.0
    best_model_dev = None
    best_h_params_dev = None
    best_metric_test = -1.0
    best_model_test = None
    best_h_params_test = None
    best_model_name_test =""
    best_model_train_name=""
    best_model_dev_name=""
    metric_list_train = []
    metric_list_dev = []
    metric_list_test = []
    clf_list = []
    for cur_h_params in h_param_comb:
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)
        clf_list.append(str(clf))
        clf.fit(x_train, y_train)
        # Predict the value of the digit on the test subset
        pred_dev = clf.predict(x_dev)
        acc_dev = metric(y_dev, pred_dev)
        pred_test = clf.predict(x_test)
        acc_test = metric(y_test, pred_test)
        pred_train = clf.predict(x_train)
        acc_train = metric(y_train, pred_train)

        metric_list_train.append(acc_train)
        metric_list_dev.append(acc_dev)
        metric_list_test.append(acc_test)

        if acc_dev > best_metric_dev:
            best_metric_dev = acc_dev
            best_model_dev_name = str(clf)
            best_model_dev = clf
            best_h_params_dev = cur_h_params
        if acc_train > best_metric_train:
            best_metric_train = acc_train
            best_model_train = clf
            best_model_train_name = str(clf)
            best_h_params_train = cur_h_params
        if acc_test > best_metric_test:
            best_metric_test = acc_test
            best_model_test = clf
            best_model_name_test = str(clf)
            best_h_params_test = cur_h_params

    df = pd.DataFrame({'Classifier':clf_list, 'train_accuracy': metric_list_train, 'dev_accuracy': metric_list_dev,'test_accuracy': metric_list_test})
    print(df.to_string())
    print("\n\t* Min, max, mean, median of the accuracies obtained in previous step : *\t\n")
    min_list = []
    max_list = []
    mean_list = []
    test_data = []
    test_data.append("Train Set")
    min_list.append(df['train_accuracy'].min())
    max_list.append(df['train_accuracy'].max())
    mean_list.append(df['train_accuracy'].mean())
    test_data.append("Dev Set")
    min_list.append(df['dev_accuracy'].min())
    max_list.append(df['dev_accuracy'].max())
    mean_list.append(df['dev_accuracy'].mean())
    test_data.append("Test Set")
    min_list.append(df['test_accuracy'].min())
    max_list.append(df['test_accuracy'].max())
    mean_list.append(df['test_accuracy'].mean())
    df_mertices = pd.DataFrame({'Data Set':test_data,'Min Accuracy':min_list,'Max Accuracy':max_list,'Mean Accuracy':mean_list})
    print(df_mertices.to_string())
    print(f"\nBest Classification Train Accuracy for classifier {best_model_train_name} is {best_metric_train:.2f}")
    print(f"\nBest Classification Dev Accuracy for classifier {best_model_dev_name} is {best_metric_dev:.2f}")
    print(f"\nBest Classification Test Accuracy for classifier {best_model_name_test} is {best_metric_test:.2f}")
    best_model_test.set_params(**best_h_params_test)
    return best_model_test, best_metric_test, best_h_params_test


def visualize_pred_data(X_test, predicted):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def generate_h_param_comb(gamma_list, c_list):
    return [{"gamma": g, "C": c} for g in gamma_list for c in c_list]