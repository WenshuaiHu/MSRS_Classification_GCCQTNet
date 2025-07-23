# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:17:47 2017

@author: acer
"""
import numpy as np
from sklearn.metrics import confusion_matrix ## 0.24.2
def confusion_matrix_image(y_true, y_pred,num_classes):
#y_true=[2,1,3,1,2,0]
#y_pred=[2,3,3,3,2,1]
    labels = list(set(y_true))
#    labels = list(num_classes) 
#    for i in range(num_classes):
#        labels(i) = i
    confusion = confusion_matrix(y_true, y_pred)
    c_len = num_classes 
    ea_acc =np.zeros(c_len)
    for j  in range(c_len):
        ea_acc[j] = confusion[j,j]/sum(confusion[j,:]);
    oa_acc = sum(np.diag((confusion)))/len(y_true);
    aa_acc = np.mean(ea_acc)
    CM = confusion.T
    kappa = (oa_acc - np.sum(np.sum(CM,axis=0) * np.sum(CM,axis=1) )/sum(sum(CM))**2) / (1-sum(np.sum(CM,axis=0) * np.sum(CM,axis=1) )/sum(sum(CM))**2)
    return confusion, oa_acc, ea_acc,aa_acc, kappa