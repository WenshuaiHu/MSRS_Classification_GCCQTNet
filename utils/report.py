import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
#import intertools

#import wandb
import random
import torch


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    ''''
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True#'''  


def compute_metrics(prediction, target, n_classes):
    """Compute and print metrics (OA, PA, AA, Kappa)

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes: number of classes, max(target) by default
        ignored_labels: list of labels to ignore, e.g. for undef

    Returns:
        {Confusion Matrix, OA, PA, AA, Kappa}
    """

    results = {}

    # compute Overall Accuracy
    cm = confusion_matrix(target, prediction, labels=range(n_classes + 1))
    results['Confusion matrix'] = cm

    # compute Overall Accuracy (OA)
    oa = 1. * np.trace(cm) / np.sum(cm)
    results['OA'] = oa

    # compute Producer Accuracy (PA)
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) for i in range(n_classes + 1)])
    results['PA'] = pa

    # compute Average Accuracy (AA)
    aa = np.mean(pa)
    results['AA'] = aa *100

    # compute kappa coefficient
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm))
    kappa = (oa - pe) / (1 - pe)
    results['Kappa'] = kappa

    return results#'''
    
'''
def compute_metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    
    #ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    #for l in ignored_labels:
        #ignored_mask[target == l] = True
    #ignored_mask = ~ignored_mask#
    #target = target[ignored_mask] -1
    # target = target[ignored_mask]
    # prediction = prediction[ignored_mask]#

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes+1))
    results["Confusion matrix"] = cm

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    results["TPR"] = TPR
    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    # compute Overall Accuracy (OA)
    results["OA"] = accuracy### OA
    
    # compute Producer Accuracy (PA)
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) for i in range(n_classes + 1)])
    results['PA'] = pa

    # compute Average Accuracy (AA)
    aa = np.mean(pa)
    results['AA'] = aa

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    results["prediction"] = prediction
    results["label"] = target

    return results#'''

''' 
def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
   
def compute_metrics(prediction, target, n_classes):
    """Compute and print metrics (OA, PA, AA, Kappa)

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes: number of classes, max(target) by default
        ignored_labels: list of labels to ignore, e.g. for undef

    Returns:
        {Confusion Matrix, OA, PA, AA, Kappa}
    """
    
    results = {}

    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    #classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(target, prediction)
    cm = confusion_matrix(target, prediction)
    each_acc, aa = AA_andEachClassAccuracy(cm)
    kappa = cohen_kappa_score(target, prediction)
    results['Confusion matrix'] = cm
    results['OA'] = oa
    results['PA'] = each_acc
    results['AA'] = aa *100
    results['Kappa'] = kappa

    return results#'''
def metrics2text(results, labels_text, replica=0):
    cm = results["Confusion matrix"]
    oa = results['OA']
    pa = results['PA']
    aa = results['AA']
    kappa = results['Kappa']

    text = ""
    text += "**Overall Accuracy**: {:.04f}  \n".format(oa)
    text += "  \n"
    text += "**Producer's Accuracy**:  \n"
    for label, acc in zip(labels_text, pa):
        text += "{}: {:.04f}  \n".format(label.center(13), acc)
    text += "  \n"
    text += "**Average Accuracy**: {:.04f}  \n".format(aa)
    text += "  \n"
    text += "**Kappa**: {:.04f}  \n".format(kappa)
    text += "  \n"

    # Console output
    text = ""
    text += "Confusion matrix:\n"
    text += str(cm)
    text += "\n---\n"

    text += "Overall Accuracy: {:.04f}\n".format(oa)
    text += "---\n"

    text += "Producer's Accuracy:\n"
    for label, acc in zip(labels_text, pa):
        text += "\t{}: {:.04f}\n".format(label, acc)
    text += "---\n"

    text += "Average Accuracy: {:.04f}\n".format(aa)
    text += "---\n"

    text += "Kappa: {:.04f}\n".format(kappa)
    text += "---\n"

    print(text)
    # wandb.log({"OA": oa, "AA": aa, "Kappa":kappa})

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap=plt.cm.Blues):
    
    #https://blog.csdn.net/qq_32425195/article/details/101537049
    
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('Normilized Confusion Matrix')
    else:
        print('Confusion Matrix without Normilization')
    #print(cm)
	
    fig = plt.figure()
	
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)##X轴字体倾斜45°
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            info = int(cm[j, i])
            plt.text(i, j, info,
                verticalalignment='center',
                horizontalalignment='center',
                color='white' if cm[i,j]>thresh else 'black')
            
    plt.tight_layout()
    #plt.show()
    #plt.close()
	
    return fig#'''
'''
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap=plt.cm.Blues):
    
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('Normilized Confusion Matrix')
    else:
        print('Confusion Matrix without Normilization')
    #print(cm)

    fig = plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.ylim(0,10)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.tight_layout()
    plt.xlable('Predicted Label')
    plt.ylable('True Label')
    plt.show()#'''
    
def cm_viz(writer, cm, labels_text, replica):
    #print('0000000000000000000000000000000000', cm)
    writer.add_figure('Confusion matrix', figure = plot_confusion_matrix(cm, classes = labels_text, normalize=False, title = 'Confusion matrix'), global_step=1)
    #writer.add_figure('Confusion matrix', figure = plot_confusion_matrix(cm, classes = labels_text, normalize=True, title = 'Confusion matrix'), global_step=1)
    '''
    test = wandb.init(project = "MS_Classification", resume = "allow")
    cm = pd.DataFrame(data=cm / np.sum(cm, axis=1, keepdims=True), index=labels_text, columns=labels_text)
    plt.figure(figsize=(12, 7))
    Img = wandb.Image(sns.heatmap(data=cm, annot=True).get_figure(), caption=f"Confusion Matrix {replica}")
    test.log({"Confusion Matrix": Img})#'''
    

def show_statistics(statistics, labels_text):
    OAs = np.array([statistics[rep]['OA'] for rep in range(len(statistics))])
    PAs = np.array([statistics[rep]['PA'] for rep in range(len(statistics))])
    AAs = np.array([statistics[rep]['AA'] for rep in range(len(statistics))])
    Kappas = np.array([statistics[rep]['Kappa'] for rep in range(len(statistics))])

    # Console output
    text = "===== Summary =====\n"

    text += "Producer's Accuracy:\n"
    for label, acc in zip(labels_text, PAs.T):
        text += "\t{}: {:.04f} ± {:.04f}\n".format(label, acc.mean(), acc.std())
    text += "---\n"

    text += f"Overall Accuracy: {OAs.mean():.04f} ± {OAs.std():.04f}\n"
    text += "---\n"

    text += f"Average Accuracy: {AAs.mean():.04f} ± {AAs.std():.04f}\n"
    text += "---\n"

    text += f"Kappa: {Kappas.mean():.04f} ± {Kappas.std():.04f}\n"
    text += "---\n"

    print(text)

    #wandb.log({"OA": OAs.mean(), "AA": AAs.mean(), "Kappa":Kappas.mean()})
