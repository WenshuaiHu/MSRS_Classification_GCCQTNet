import scipy.io as scio
import numpy as np
from .data_loader_train import applyPCA
def addZeroPadding(X, margin=2):
    """
    add zero padding to the image
    """
    newX = np.zeros((
      X.shape[0] + 2 * margin,
      X.shape[1] + 2 * margin,
      X.shape[2]
            ))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)
    
def data_loader(args):
    # if args.dataset == 'Berlin':
      # args.hsi_bands = 244
      # args.sar_bands = 4
      # args.num_class = 8
    # elif args.dataset == 'Augsburg':
      # args.hsi_bands = 180
      # args.sar_bands = 4
      # args.num_class = 7
    # elif args.dataset == 'HHK':
      # args.hsi_bands = 166
      # args.sar_bands = 3
      # args.num_class = 5
    # elif args.dataset == 'Muufl':
        # args.hsi_bands = 64
        # args.sar_bands = 2
        # args.num_class = 11
    # elif args.dataset == 'Houston':
        # args.hsi_bands = 144
        # args.sar_bands = 2
        # args.num_class = 15

    if args.dataset == "Houston":
        data_hsi = scio.loadmat(args.root + args.dataset + '/data_hsi.mat')['houston_hsi']
        data_sar = scio.loadmat(args.root + args.dataset + '/data_sar.mat')['houston_lidar']
        data_gt = scio.loadmat(args.root + args.dataset + '/mask_test.mat')['mask_test']
    elif args.dataset == "Trento":
        data_hsi = scio.loadmat(args.root + args.dataset + '/trento_hsi.mat')['trento_hsi']
        data_sar = scio.loadmat(args.root + args.dataset + '/trento_lidar.mat')['trento_lidar']
        #data_sar = data_sar[:,:,0]
        data_gt = scio.loadmat(args.root + args.dataset + '/mask_test.mat')['mask_test']
    else: 
        data_hsi = scio.loadmat(args.root + args.dataset + '/data_hsi.mat')['data']
        data_sar = scio.loadmat(args.root + args.dataset + '/data_sar.mat')['data']
        #data_sar = data_sar[:,:,0]
        data_gt = scio.loadmat(args.root + args.dataset + '/mask_test.mat')['mask_test']

    test_mask = data_gt
    
    if data_sar.ndim==2:
        data_sar = np.expand_dims(data_sar, axis=2)
        #data_sar = data_sar.repeat(2, axis=2)
        
    height, width, c = data_hsi.shape
    data_hsi = minmax_normalize (data_hsi)
    data_sar = minmax_normalize(data_sar)
    data_hsi,pca = applyPCA(data_hsi,numComponents=args.hsi_bands)

    margin = (args.patch_size-1) // 2
    data_hsi = addZeroPadding(data_hsi, margin)
    data_sar = addZeroPadding(data_sar, margin)
    data_gt = np.pad(data_gt, ((margin, margin), (margin, margin)), 'constant', constant_values=(0, 0))
    
    test_loader = {}
    
    test_loader['data_hsi'] = data_hsi
    test_loader['data_sar'] = data_sar
    test_loader['data_gt'] = data_gt
    test_loader['mask_test'] = test_mask
    test_loader['margin'] = margin#'''
    '''
    test_loader = {'data_hsi':data_hsi, 
                'data_sar':data_sar, 
                'data_gt':data_gt, 
                'mask_test':test_mask, 
                'margin': margin}#'''
    
    
    return test_loader
    
