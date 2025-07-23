import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def get_dataset_info(dataset_name):
    if dataset_name == "Muufl":
        labels_text = ['Trees', 'Grass ground', 'Mixed ground', 'Dirt and sand', 'Road', 'Water', 'Buildings', 'Shadow', 'Sidewalk', 'Yellow curb', 'Cloth panels']
        ignored_label = -1
    elif dataset_name == "Trento":
        labels_text = ['Apple trees', 'Buildings', 'Ground', 'Woods', 'Vineyard', 'Roads']
        ignored_label = -1
    elif dataset_name == "Houston":
        labels_text = ['Health grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track']
        ignored_label = -1
    elif dataset_name == "Augsburg":
        labels_text = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Allotment', 'Commercial Area', 'Water']
        ignored_label = -1
    elif dataset_name == "Berlin":
        labels_text = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment', 'Commercial Area', 'Water']
        ignored_label = -1
		
    else:
        raise ValueError("Dataset must be one of MUUFL, Trento, Houston, Augsburg, Berlin")
    return labels_text



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

def createImgCube(X ,gt ,pos:list ,windowSize=25):
    """
    create Cube from pos list
    return imagecube gt nextPos
    """    
    margin = (windowSize-1)//2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    dataPatches = np.zeros((pos.__len__(), windowSize, windowSize, X.shape[2]))
    if( pos[-1][1]+1 != X.shape[1] ):
        nextPos = (pos[-1][0] ,pos[-1][1]+1)
    elif( pos[-1][0]+1 != X.shape[0] ):
        nextPos = (pos[-1][0]+1 ,0)
    else:
        nextPos = (0,0)
    return np.array([zeroPaddingX[i:i+windowSize, j:j+windowSize, :] for i,j in pos ]),\
    np.array([gt[i,j] for i,j in pos]) ,\
    nextPos

def createPos(shape:tuple, pos:tuple, num:int):
    """
    creatre pos list after the given pos
    """
    if (pos[0]+1)*(pos[1]+1)+num >shape[0]*shape[1]:
        num = shape[0]*shape[1]-( (pos[0])*shape[1] + pos[1] )
    return [(pos[0]+(pos[1]+i)//shape[1] , (pos[1]+i)%shape[1] ) for i in range(num) ]

def createPosWithoutZero(hsi, gt):
    """
    creatre pos list without zero labels
    """
    mask = gt > 0
    return [(i,j) for i , row  in enumerate(mask) for j , row_element in enumerate(row) if row_element]

def splitTrainTestSet(X, gt, testRatio, randomState=111):
    """
    random split data set
    """
    X_train, X_test, gt_train, gt_test = train_test_split(X, gt, test_size=testRatio, random_state=randomState, stratify=gt)
    return X_train, X_test, gt_train, gt_test

def createImgPatch(lidar, pos:list, windowSize=25):
    """
    return lidar Img patches
    """
    margin = (windowSize-1)//2
    zeroPaddingLidar = np.zeros((
      lidar.shape[0] + 2 * margin,
      lidar.shape[1] + 2 * margin
            ))
    zeroPaddingLidar[margin:lidar.shape[0]+margin, margin:lidar.shape[1]+margin] = lidar
    return np.array([zeroPaddingLidar[i:i+windowSize, j:j+windowSize] for i,j in pos ])

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def data_aug(train_hsiCube, train_patches, train_labels):
    Xh = []
    Xl = []
    y = []
    for i in range(train_hsiCube.shape[0]):
        Xh.append(train_hsiCube[i])
        Xl.append(train_patches[i])

        noise = np.random.normal(0.0, 0.02, size=train_hsiCube[0].shape)
        noise2 = np.random.normal(0.0, 0.02, size=train_patches[0].shape)
        Xh.append(np.flip(train_hsiCube[i] + noise, axis=1))
        Xl.append(np.flip(train_patches[i] + noise2, axis=1))

        k = np.random.randint(4)
        Xh.append(np.rot90(train_hsiCube[i], k=k))
        Xl.append(np.rot90(train_patches[i], k=k))
        # Xh.append(np.rot90(train_hsiCube[i]))
        # Xl.append(np.rot90(train_patches[i]))
        
        # Xh.append(np.rot90(np.rot90(train_hsiCube[i])))
        # Xl.append(np.rot90(np.rot90(train_patches[i])))
        
        # Xh.append(np.rot90(np.rot90(np.rot90(train_hsiCube[i]))))
        # Xl.append(np.rot90(np.rot90(np.rot90(train_patches[i]))))
        
        
        # y.append(train_labels[i])
        # y.append(train_labels[i])
        y.append(train_labels[i])
        y.append(train_labels[i])
        y.append(train_labels[i])

    train_labels = np.asarray(y, dtype=np.int8)
    train_hsiCube = np.asarray(Xh, dtype=np.float32)
    train_patches = np.asarray(Xl, dtype=np.float32)
    train_hsiCube = torch.from_numpy(train_hsiCube.transpose(0, 3, 1, 2)).float()
    train_patches = torch.from_numpy(train_patches.transpose(0, 3, 1, 2)).float()
    return train_hsiCube, train_patches, train_labels

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, hsi, sar, labels):
        self.len = labels.shape[0]
        self.hsi = torch.FloatTensor(hsi)
        self.sar = torch.FloatTensor(sar)
        self.labels = torch.LongTensor(labels - 1)
    def __getitem__(self, index):
        return self.hsi[index], self.sar[index], self.labels[index]
    def __len__(self):
        return self.len

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca
        
def build_datasets(args):
    
    if args.dataset == "Houston":
        data_hsi = scio.loadmat(args.root + args.dataset + '/data_hsi.mat')['houston_hsi']
        data_sar = scio.loadmat(args.root + args.dataset + '/data_sar.mat')['houston_lidar']
        data_traingt = scio.loadmat(args.root + args.dataset + '/mask_train.mat')['mask_train']
        #print(1111111111, args.dataset)
    elif args.dataset == "Trento":
        data_hsi = scio.loadmat(args.root + args.dataset + '/trento_hsi.mat')['trento_hsi']
        data_sar = scio.loadmat(args.root + args.dataset + '/trento_lidar.mat')['trento_lidar']
        #data_sar = data_sar[:,:,0]
        data_traingt = scio.loadmat(args.root + args.dataset + '/mask_train.mat')['mask_train']
    else: 
        data_hsi = scio.loadmat(args.root + args.dataset + '/data_hsi.mat')['data']
        data_sar = scio.loadmat(args.root + args.dataset + '/data_sar.mat')['data']
        #data_sar = data_sar[:,:,0]
        #print(111111111, data_sar)
        data_traingt = scio.loadmat(args.root + args.dataset + '/mask_train.mat')['mask_train']

    if data_sar.ndim==2:
        data_sar = np.expand_dims(data_sar, axis=2)
        #data_sar = data_sar.repeat(2, axis=2)
    print(data_hsi.shape, data_sar.shape)
    data_hsi = minmax_normalize(data_hsi)
    data_sar = minmax_normalize(data_sar)
    
    data_hsi,pca = applyPCA(data_hsi,numComponents=args.hsi_bands)
    
    # training / testing set for 2D-CNN
    train_hsiCube, train_labels ,_ = createImgCube(data_hsi, data_traingt, createPosWithoutZero(data_hsi, data_traingt), windowSize=args.patch_size)
    train_patches, _ ,_ = createImgCube(data_sar, data_traingt, createPosWithoutZero(data_sar, data_traingt), windowSize=args.patch_size)

    train_hsiCube, train_patches, train_labels = data_aug(train_hsiCube, train_patches, train_labels)
    X_train, X_test, gt_train, gt_test = splitTrainTestSet(train_hsiCube, train_labels, args.test_ratio, randomState=128)
    X_train_2, X_test_2, _, _ = splitTrainTestSet(train_patches, train_labels, args.test_ratio, randomState=128)
    ## X_train HSI；X_train_2 SAR
    print (X_train.shape)
    print (X_test.shape)
    print("Creating dataloader")
    
    ###  delete the validation
    # X_train = np.concatenate((X_train, X_test), axis=0)
    # gt_train = np.concatenate((gt_train, gt_test), axis=0)
    # X_train_2 = np.concatenate((X_train_2, X_test_2), axis=0)
    # print (X_train.shape)
    # print (X_test.shape)#'''
    
    classes = gt_train.max()
    ld = dict()
    for i in range(classes):
        ld[i]=0
    for j in range(len(gt_train)):
        label = gt_train[j]-1
        ld[label] += 1
    weight2class = ld.values()
    
    
    #trainset = TensorDataset(X_train, X_train_2, gt_train)
    trainset = TensorDataset(train_hsiCube, train_patches, train_labels)
    
    testset = TensorDataset(X_test, X_test_2, gt_test)
    train_loader = torch.utils.data.DataLoader(dataset= trainset, batch_size= args.batch_size, shuffle= True, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(dataset= testset, batch_size= args.batch_size, shuffle= False, num_workers = 0)

    return train_loader, test_loader