import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from mri_util import print_execution_time
from datetime import datetime
def mri_metric(label_dir, pred_dir):
    execute_date = datetime.now().strftime("%Y%m%d")
    os.makedirs(f"./result/{execute_date}", exist_ok = True)

    label_file = pd.read_csv(label_dir)
    pred_file = pd.read_csv(pred_dir)

    pred_file = pred_file[pred_file["pred"] != 'Error']
    pred_file = pred_file[["path", "pred"]]
    label_file["filepath"] = label_file["filepath"].apply(lambda x: '/'.join(x.split('/')[4:]))
    pred_file["path"] = pred_file["path"].apply(lambda x: '/'.join(x.split('/')[4:]))
    
    label_file.sort_values("filepath")
    pred_file.sort_values("path")

    result = pd.merge(label_file, pred_file, left_on="filepath", right_on = "path", how = "inner")
    result.to_csv(f"result/{execute_date}/mri_result.csv", index = False)

    label = result[["label"]]
    pred = result[["pred"]]

    accuracy = accuracy_score(label, pred)
    recall = recall_score(label, pred, average = 'weighted')
    precision = precision_score(label, pred, average = 'weighted')

    specificities = []

    num_classes = len(np.unique(pred))
    encoder = LabelEncoder()
    actual_encoded = encoder.fit_transform(label)
    predicted_encoded = encoder.transform(pred)
    for i in range(num_classes):
        tn = np.sum((np.array(actual_encoded) != i) & (np.array(predicted_encoded) != i))
        fp = np.sum((np.array(actual_encoded) != i) & (np.array(predicted_encoded) == i))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    metric_report = pd.DataFrame([[accuracy, recall, precision, sum(specificities) / len(specificities)]], columns = ["accuracy", "recall", "precision", "specificity"])
    metric_report.to_csv(f"result/{execute_date}/mri_metric.csv", index = False)


def xray_metric(label_dir, pred_dir):
    execute_date = datetime.now().strftime("%Y%m%d")
    os.makedirs(f"./result/{execute_date}", exist_ok = True)

    label_file = pd.read_csv(label_dir)
    # label_file = label_file[label_file["modality"] == 'CR']
    label_file = label_file[["filepath", "label"]]
    label_file.sort_values("filepath")
    label_file["filepath"] = label_file["filepath"].apply(lambda x: x.split('/')[-1])
    # label_file.loc[label_file["label"] == "others", "label"] = "Others"

    pred_file = pd.read_csv(pred_dir)
    pred_file.sort_values("file")
    pred_file = pred_file[["file", "pred"]]

    result = pd.merge(label_file, pred_file, left_on="filepath", right_on = "file", how = "inner")
    result.to_csv(f"result/{execute_date}/xray_result.csv", index = False)

    label = result[["label"]]
    pred = result[["pred"]]

    num_classes = len(np.unique(pred))

    accuracy = accuracy_score(label, pred)
    recall = recall_score(label, pred, average = 'weighted')
    precision = precision_score(label, pred, average = 'weighted')

    specificities = []
    
    encoder = LabelEncoder()
    actual_encoded = encoder.fit_transform(label)
    predicted_encoded = encoder.transform(pred)
    for i in range(num_classes):
        tn = np.sum((np.array(actual_encoded) != i) & (np.array(predicted_encoded) != i))
        fp = np.sum((np.array(actual_encoded) != i) & (np.array(predicted_encoded) == i))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    metric_report = pd.DataFrame([[accuracy, recall, precision, sum(specificities) / len(specificities)]], columns = ["accuracy", "recall", "precision", "specificity"])
    metric_report.to_csv(f"result/{execute_date}/xray_metric.csv", index = False)

if __name__ == "__main__":
    start_time = datetime.now()
    # mri_label_dir = "./label/mri_label.csv"
    # mri_pred_dir = "./data/MR/result.csv"
    # mri_metric(mri_label_dir, mri_pred_dir)

    xray_label_dir = "./label/xray_label.csv"
    xray_pred_dir = "./data/CR/result.csv"
    xray_metric(xray_label_dir, xray_pred_dir)


    print_execution_time(start_time, 'calc_metric')




