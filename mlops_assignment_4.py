
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score

from utilpy import data_preprocess,h_param_tuning,train_dev_test_split

digits = datasets.load_digits()

data, label = data_preprocess(digits)

del digits

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.6, 1, 2, 5, 7, 10]
h_param_comb_svm = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

min_samples_split_list = [2,3,5,10]
min_samples_leaf_list = [1,3,5,10]
h_param_comb_dtree = [{"min_samples_leaf": g, "min_samples_split": c} for g in min_samples_leaf_list for c in min_samples_split_list]

model_of_choices = [svm.SVC(),tree.DecisionTreeClassifier()]
hp_of_choices = [h_param_comb_svm,h_param_comb_dtree]
metric=metrics.accuracy_score

acuu_list_List = []
f1_mac_ll = []
f1_mic_ll = []
f1_wt_ll = []

pre_mac_ll = []
pre_mic_ll = []
pre_wt_ll = []


rec_mac_ll = []
rec_mic_ll = []
rec_wt_ll = []
z_l = []
x = list(set(label))

for k in range(5):
    model_list = []
    accu_list = []
    f1_score_macro = []
    f1_score_micro = []
    f1_score_weighted = []
    recall_score_macro = []
    recall_score_micro = []
    recall_score_weighted = []
    precision_score_macro = []
    precision_score_micro = []
    precision_score_weighted = []
    f1_score_l = []
    for i,clf in enumerate(model_of_choices):

        X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
            data, label, 0.7, 0.1
        )

        best_model, best_metric, best_h_params = h_param_tuning(hp_of_choices[i], clf, X_train, y_train, X_dev, y_dev, X_test, y_test, metric)
        prediction = best_model.predict(X_test)
        model_list.append(best_model)
        accu_list.append(accuracy_score(y_test, prediction))
        cm = metrics.confusion_matrix(y_test, prediction)
        cm_nor = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        f1_score_l.append(list(cm_nor.diagonal()))
        f1_score_macro.append(f1_score(y_test, prediction,average="macro"))
        f1_score_micro.append(f1_score(y_test, prediction, average="micro"))
        f1_score_weighted.append(f1_score(y_test, prediction, average="weighted"))
        recall_score_micro.append(recall_score(y_test,prediction,average="micro"))
        recall_score_macro.append(recall_score(y_test, prediction, average="macro"))
        recall_score_weighted.append(recall_score(y_test, prediction, average="weighted"))
        precision_score_micro.append(precision_score(y_test,prediction,average="micro"))
        precision_score_macro.append(precision_score(y_test, prediction, average="macro"))
        precision_score_weighted.append(precision_score(y_test, prediction, average="weighted"))

        print("\n")
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, prediction)}\n"
        )
        print(
            f"Confusion report for classifier {clf}:\n"
            f"{metrics.confusion_matrix(y_test, prediction)}\n"
        )

    z_l.append(f1_score_l)
    acuu_list_List.append(accu_list)
    f1_wt_ll.append(f1_score_weighted)
    f1_mic_ll.append(f1_score_micro)
    f1_mac_ll.append(f1_score_macro)

    pre_mic_ll.append(precision_score_micro)
    pre_mac_ll.append(precision_score_macro)
    pre_wt_ll.append(precision_score_weighted)

    rec_wt_ll.append(recall_score_weighted)
    rec_mac_ll.append(recall_score_macro)
    rec_mic_ll.append(recall_score_micro)

colours=[['r','g'],['b','k'],['c','m'],['y','r'],['g','b']]

fig, axes = plt.subplots(5)
fig.suptitle('Class Level Accuracy Trend for SVC & DT')
plt.xticks(x)
for i in range(len(z_l)):
    thl = z_l[i]
    y1 = thl[0]
    y2 = thl[1]
    thisClr = colours[i]
    axes[i].plot(x, y1,thisClr[0],label="SVC",marker='o')
    axes[i].plot(x, y2,thisClr[1], label="Decision Tree",marker='o')
    axes[i].legend()

for ax in fig.get_axes():
    ax.label_outer()
plt.show()


df_acc = pd.DataFrame(acuu_list_List,columns=["SVC","Decision Tree"])
print("Accuracy Comparison of two Models")
print("-----------------------------------")
print(df_acc.to_string())
print(f"Mean {round(df_acc['SVC'].mean(),3)} \t\t {round(df_acc['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_acc['SVC'].std(),3)} \t\t {round(df_acc['Decision Tree'].std(),3)} ")

print("\n")
print("F1 Score(Macro) Comparison of two Models")
print("-----------------------------------------")
df_f1_mac = pd.DataFrame(f1_mac_ll,columns=["SVC","Decision Tree"])
print(df_f1_mac.to_string())
print(f"Mean {round(df_f1_mac['SVC'].mean(),3)} \t\t {round(df_f1_mac['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_f1_mac['SVC'].std(),3)} \t\t {round(df_f1_mac['Decision Tree'].std(),3)} ")

print("\n")
print("F1 Score(Micro) Comparison of two Models")
print("--------------------------------------------")
df_f1_mic = pd.DataFrame(f1_mic_ll,columns=["SVC","Decision Tree"])
print(df_f1_mic.to_string())
print(f"Mean {round(df_f1_mic['SVC'].mean(),3)} \t\t {round(df_f1_mic['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_f1_mic['SVC'].std(),3)} \t\t {round(df_f1_mic['Decision Tree'].std(),3)} ")

print("\n")
print("F1 Score(Weighted) Comparison of two Models")
print("----------------------------------------")
df_f1_wt = pd.DataFrame(f1_wt_ll,columns=["SVC","Decision Tree"])
print(df_f1_wt.to_string())
print(f"Mean {round(df_f1_wt['SVC'].mean(),3)} \t\t {round(df_f1_wt['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_f1_wt['SVC'].std(),3)} \t\t {round(df_f1_wt['Decision Tree'].std(),3)} ")

print("\n")
print("Precision Score(Macro) Comparison of two Models")
print("-----------------------------------------------")
df_pre_mac = pd.DataFrame(pre_mac_ll,columns=["SVC","Decision Tree"])
print(df_pre_mac.to_string())
print(f"Mean {round(df_pre_mac['SVC'].mean(),3)} \t\t {round(df_pre_mac['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_pre_mac['SVC'].std(),3)} \t\t {round(df_pre_mac['Decision Tree'].std(),3)} ")

print("\n")
print("Precision Score(Micro) Comparison of two Models")
print("---------------------------------------------------")
df_pre_mic = pd.DataFrame(pre_mic_ll,columns=["SVC","Decision Tree"])
print(df_pre_mic.to_string())
print(f"Mean {round(df_pre_mic['SVC'].mean(),3)} \t\t {round(df_pre_mic['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_pre_mic['SVC'].std(),3)} \t\t {round(df_pre_mic['Decision Tree'].std(),3)} ")

print("\n")
print("Precision Score(Weighted) Comparison of two Models")
print("------------------------------------------------")
df_pre_wt = pd.DataFrame(pre_wt_ll,columns=["SVC","Decision Tree"])
print(df_pre_wt.to_string())
print(f"Mean {round(df_pre_wt['SVC'].mean(),3)} \t\t {round(df_pre_wt['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_pre_wt['SVC'].std(),3)} \t\t {round(df_pre_wt['Decision Tree'].std(),3)} ")

print("\n")
print("Recall Score(Macro) Comparison of two Models")
print("--------------------------------------------")
df_rec_mac = pd.DataFrame(rec_mac_ll,columns=["SVC","Decision Tree"])
print(df_rec_mac.to_string())
print(f"Mean {round(df_rec_mac['SVC'].mean(),3)} \t\t {round(df_rec_mac['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_rec_mac['SVC'].std(),3)} \t\t {round(df_rec_mac['Decision Tree'].std(),3)} ")

print("\n")
print("Recall Score(Micro) Comparison of two Models")
print("------------------------------------------------")
df_rec_mic = pd.DataFrame(rec_mic_ll,columns=["SVC","Decision Tree"])
print(df_rec_mic.to_string())
print(f"Mean {round(df_rec_mic['SVC'].mean(),3)} \t\t {round(df_rec_mic['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_rec_mic['SVC'].std(),3)} \t\t {round(df_rec_mic['Decision Tree'].std(),3)} ")

print("\n")
print("Recall Score(Weighted) Comparison of two Models")
print("--------------------------------------------")
df_rec_wt = pd.DataFrame(rec_wt_ll,columns=["SVC","Decision Tree"])
print(df_rec_wt.to_string())
print(f"Mean {round(df_rec_wt['SVC'].mean(),3)} \t\t {round(df_rec_wt['Decision Tree'].mean(),3)} ")
print(f"Std {round(df_rec_wt['SVC'].std(),3)} \t\t {round(df_rec_wt['Decision Tree'].std(),3)} ")

