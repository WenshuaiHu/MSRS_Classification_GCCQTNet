

import torch
import numpy as np
import time
import os
#from utils.confusion_matrix_image import confusion_matrix_image
from utils.report import compute_metrics
  
def metrics(args, y_true, y_pred, num_classes, result_writer, time_writer, testing_time, epoch):

    with open (time_writer, 'a') as  f_timecal:                                                                                      
        f_timecal.write(str(testing_time)+'\n')##y_true.cpu().numpy
        f_timecal.close()
		
    '''
    [confusion, oa_acc,aa_acc,aa_acc_all, kappa] = confusion_matrix_image(y_true, y_pred, num_classes)
    print('*****************', oa_acc)
    
    #oa_acc = oa_acc.astype(np.str)
    #aa_acc = aa_acc.astype(np.str)
    #aa_acc_all = aa_acc_all.astype(np.str)
    #kappa = kappa.astype(np.str)#
    
    oa_acc = str(oa_acc)
    aa_acc = str(aa_acc)
    aa_acc_all = str(aa_acc_all)
    kappa = str(kappa)#'''
	
    results = compute_metrics(
            prediction=y_pred, #pred[data.test_mask].detach().cpu().numpy(),
            target=y_true, #data.y[data.test_mask].detach().cpu().numpy(),
            n_classes=int(y_true.max())
    )###
    #print (' *******************************', int(y_pred.max()), int(y_true.max()))# # 此次需类别是从0-7 8个类别
	
    aa_acc_all, oa_acc, kappa, aa_acc = results['AA'], results['OA'], results['Kappa'], results['PA']
    #print(aa_acc_all)
	
    #Houston 
    if args.dataset=='Houston':
        result_writer.write(str(epoch+1) + ',' + str(oa_acc) + ',' + \
                         str(aa_acc[0]) + ',' + str(aa_acc[1]) + ',' + str(aa_acc[2]) + \
                         ',' + str(aa_acc[3]) + ',' + str(aa_acc[4]) + ',' + str(aa_acc[5]) \
                         + ',' + str(aa_acc[6]) + ',' + str(aa_acc[7]) + ',' + str(aa_acc[8]) \
                         + ',' + str(aa_acc[9]) + ',' + str(aa_acc[10]) + ',' + str(aa_acc[11]) \
                         + ',' + str(aa_acc[12]) + ',' + str(aa_acc[13]) + ',' + str(aa_acc[14]) \
                         + ',' + str(aa_acc_all) +',' + str(kappa) + '\n')
    #Trento 
    if args.dataset=='Trento':
        result_writer.write(str(epoch+1) + ',' + str(oa_acc) + ',' + \
                         str(aa_acc[0]) + ',' + str(aa_acc[1]) + ',' + str(aa_acc[2]) + \
                         ',' + str(aa_acc[3]) + ',' + str(aa_acc[4]) + ',' + str(aa_acc[5]) \
                         + ',' + str(aa_acc_all) +',' + str(kappa) + '\n')
    #Indian
    if args.dataset=='Berlin':
        result_writer.write(str(epoch+1) + ',' + str(oa_acc) + ',' + \
						str(aa_acc[0]) + ',' + str(aa_acc[1]) + ',' + str(aa_acc[2]) + \
						',' + str(aa_acc[3]) + ',' + str(aa_acc[4]) + ',' + str(aa_acc[5]) \
						+ ',' + str(aa_acc[6]) + ',' + str(aa_acc[7]) + ',' + str(aa_acc_all) +',' + str(kappa) + '\n')
    #Ksc             
    if args.dataset=='Augsburg':
        result_writer.write(str(epoch+1) + ',' + str(oa_acc) + ',' + \
						str(aa_acc[0]) + ',' + str(aa_acc[1]) + ',' + str(aa_acc[2]) + \
						',' + str(aa_acc[3]) + ',' + str(aa_acc[4]) + ',' + str(aa_acc[5]) \
						+ ',' + str(aa_acc[6]) + ',' + str(aa_acc_all) +',' + str(kappa) + '\n')
    #Pavia 
    if args.dataset=='pavia':
        result_writer.write(str(epoch+1) + ',' + str(oa_acc) + ',' + \
                        str(aa_acc[0]) + ',' + str(aa_acc[1]) + ',' + str(aa_acc[2]) + \
                        ',' + str(aa_acc[3]) + ',' + str(aa_acc[4]) + ',' + str(aa_acc[5]) \
                        + ',' + str(aa_acc[6]) + ',' + str(aa_acc[7]) + ',' + str(aa_acc[8]) \
                        + ',' + str(aa_acc_all) +',' + str(kappa) + '\n')
    #Salinas 					
    if args.dataset=='salinas':
        result_writer.write(str(epoch+1) + ',' + str(oa_acc) + ',' + \
						str(aa_acc[0]) + ',' + str(aa_acc[1]) + ',' + str(aa_acc[2]) + \
						',' + str(aa_acc[3]) + ',' + str(aa_acc[4]) + ',' + str(aa_acc[5]) \
						+ ',' + str(aa_acc[6]) + ',' + str(aa_acc[7]) + ',' + str(aa_acc[8]) \
						+ ',' + str(aa_acc[9]) + ',' + str(aa_acc[10]) + ',' + str(aa_acc[11]) \
						+ ',' + str(aa_acc[12]) + ',' + str(aa_acc[13]) + ',' + str(aa_acc[14]) \
						+ ',' + str(aa_acc[15]) + ',' + str(aa_acc_all) +',' + str(kappa) + '\n')
            
    if args.dataset=='Muufl':
        result_writer.write(str(epoch+1) + ',' + str(oa_acc) + ',' + \
						str(aa_acc[0]) + ',' + str(aa_acc[1]) + ',' + str(aa_acc[2]) + \
						',' + str(aa_acc[3]) + ',' + str(aa_acc[4]) + ',' + str(aa_acc[5]) \
						+ ',' + str(aa_acc[6]) + ',' + str(aa_acc[7]) + ',' + str(aa_acc[8]) \
						+ ',' + str(aa_acc[9]) + ',' + str(aa_acc[10]) \
						+ ',' + str(aa_acc_all) +',' + str(kappa) + '\n')
            
    result_writer.flush()
    return results