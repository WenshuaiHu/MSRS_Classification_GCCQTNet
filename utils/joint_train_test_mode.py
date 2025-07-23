import time
import os
import torch.optim
from time import *
import scipy.io as scio
import numpy as np
from data_loader import data_loader_test
from utils.metrics import metrics
import torchvision
import matplotlib.pyplot as plt
from utils.plt_result import plot_result
import torch.nn.functional as F

#args = args_parser.args_parser()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#print (args)

def main(model, device, test_loader, args, writer, epoch, num_result, result_writer, PATH_result, load_checkpoints=False):

    #data_hsi, data_sar, data_gt, test_mask, height, width, margin = data_loader_test.data_loader(args)
    
    data_hsi = test_loader['data_hsi']
    data_sar = test_loader['data_sar']
    data_gt = test_loader['data_gt']
    test_mask = test_loader['mask_test']
    margin = test_loader['margin']
    
    height, width, c = data_hsi.shape
    idx, idy = np.where(data_gt != 0)
    labelss = np.array([0])
    
    #print(1111111111111111111111111, data_gt.shape, data_hsi.shape)
    ### data_gt (1733, 486) (1733, 486, 244)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    if load_checkpoints:
        model = Net(args.hsi_bands, args.sar_bands, args.hidden_size, Bottleneck, args.num_parallel, args.num_reslayer,
                args.num_class, args.bn_threshold).eval().to(device)
        model.load_state_dict(torch.load('./checkpoints/model.pth'))
    else:
        model = model#'''

    batch = 200
    num = 10
    total_batch = int(len(idx)/batch +1)
    print ('Total batch number is :', total_batch, data_gt.max())
    end = time()
    for j in range(int((len(idx) - (len(idx) % batch)) / batch + 1)):
        if int(100*j // total_batch) == num:
            print('... ... ',  int(num ), '% batch handling ... ...')
            num = num + 10
        if batch * (j + 1) > len(idx):
            num_cat = len(idx) - batch * j
        else:
            num_cat = batch

        tmphsi = np.array([data_hsi[idx[j*batch + i] - margin:idx[j*batch + i] +
                                                       margin + 1, idy[j * batch + i] - margin:idy[j*batch + i] + margin + 1, :] for i in range(num_cat)])
        tmpsar = np.array([data_sar[idx[j*batch + i] - margin:idx[j*batch + i] +
                                                        margin + 1,idy[j*batch + i] - margin:idy[j*batch + i] + margin + 1,:] for i in range(num_cat)])
                                                        
        tmphsi = torch.FloatTensor(tmphsi.transpose(0, 3, 1, 2)).to(device)
        tmpsar = torch.FloatTensor(tmpsar.transpose(0, 3, 1, 2)).to(device)

        [prediction,_,_], _ = model(tmphsi, tmpsar)
        #prediction = F.softmax(prediction, dim=1)
        labelss = np.hstack([labelss, np.argmax(prediction.detach().cpu().numpy(), axis=1)])
    print('... ... ', int(100), '% batch handling ... ...')
    labelss = np.delete(labelss, [0])
    new_map = np.zeros((height, width))
    
    label_gt = np.zeros(shape=[len(idx)])
    label_gt = np.array(label_gt).astype(np.int32)
    
    label_gt_cm = np.zeros(shape=[len(idx)])
    label_gt_cm = np.array(label_gt_cm).astype(np.int32)
    
    for i in range(len(idx)):
        new_map[idx[i] - margin, idy[i] - margin] = labelss[i] + 1
        label_gt[i]= test_mask[idx[i] - margin, idy[i] - margin]
        label_gt_cm[i]= test_mask[idx[i] - margin, idy[i] - margin]-1
        
    print('Testing Finished!!!')
    end2 = time()
    minutes = int((end2 - end) / 60)
    seconds = int((end2 - end) - minutes * 60)
    print("Testing time：", minutes, "m", seconds, "s")
    
    testing_time = end2 - end
    time_writer = PATH_result + "{}".format(num_result) + '/time_cal_{}.txt'.format(args.dataset)
    
    scio.savemat(PATH_result  + "{}".format(num_result) +  '/result_{}.mat'.format(epoch+1), {'output': new_map})
    plot_result(new_map, PATH_result, num_result, epoch = epoch+1)
    
    new_map_show = torch.from_numpy(new_map).float()
    print(new_map_show.shape)
    new_map_show = torch.unsqueeze(new_map_show, 0)
    print(new_map_show.shape)
    imgs = torchvision.utils.make_grid(new_map_show)
    writer.add_image('Images', new_map_show, epoch)
	
    results = metrics(args, label_gt_cm, labelss, data_gt.max(), result_writer, time_writer, testing_time, epoch)
    return results
#if __name__ == '__main__':
    #main()
