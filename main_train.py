import os
from time import *
import torch
import torch.optim
from torch import nn
from models.Dual_channelNet import GCCQTNet
from data_loader import data_loader_train
from data_loader import data_loader_test
from utils import args_parser
import numpy as np
from utils.joint_train_test_mode import main as test_main
args = args_parser.args_parser()
from torch.nn.functional import threshold, cosine_similarity
from torch.utils.tensorboard import SummaryWriter
from utils.report import cm_viz, compute_metrics, metrics2text, show_statistics, same_seeds
from utils.write2csv import write2csv
from models.utils.config import config_2Dmf
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
def calc_loss(outputs, labels):

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss

def L1_penalty(var):
    return torch.abs(var).sum()

# def cos_similarity(z1,z2):
    # z1 = nn.functional.normalize(z1, dim=1)
    # z2 = nn.functional.normalize(z2, dim=1)
    # return 1-torch.mean(cosine_similarity(z1,z2,dim=1), dim=0)#
    
def cos_similarity(z1,z2, temp=0.1):
    loss = 0
    for i in range(len(z1)):
        # a = nn.functional.normalize(z1[i], dim=1)
        # b = nn.functional.normalize(z2[i], dim=1)
        loss += 1-torch.mean(cosine_similarity(z1[i],z2[i],dim=1), dim=0)#
    return loss# 
     
def train(model, device, train_loader, batch_size, optimizer, writer, epoch, slim_params, bn_threshold):
    model.train()
    total_loss = 0
    for i, (inputs_1, inputs_2, labels) in enumerate(train_loader):
        inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        [out_f, _, _], [z1,z2] = model(inputs_1, inputs_2)
        
        loss1 = calc_loss(out_f, labels)
        
        loss = loss1
        L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])        
        lambda_ = 0.15
        loss = loss + 0.05 * L1_norm + lambda_ * cos_similarity(z1,z2)
        #loss = loss + 0.05 * InfoNCE_loss(z1,z2, temp=0.1)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    slim_params_list = []
    for slim_param in slim_params:
       slim_params_list.extend(slim_param.cpu().data.numpy())
    slim_params_list = np.array(sorted(slim_params_list))
    #  print (slim_params, len(slim_params))
    #print('Epoch %d, 3%% smallest slim_params: %.4f' % (epoch, slim_params_list[len(slim_params_list) // 33]), flush=True, end= " ")
    print('Epoch %d, [loss avg: %.4f]   [current loss: %.4f]' %(epoch, total_loss/(epoch+1), loss.item()))
    
    writer.add_scalar('Loss/train', loss, epoch)
    
def validation(model, device, valid_loader):
    count = 0
    for inputs_1, inputs_2, labels in valid_loader:
        inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
        [outputs,_,_], _ = model(inputs_1, inputs_2)
        outputs = F.softmax(outputs, dim=1)
        
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            test_labels = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            test_labels = np.concatenate((test_labels, labels))
    a = 0
    for c in range(len(y_pred_test)):
        if test_labels[c]==y_pred_test[c]:
            a = a + 1
    acc = a/len(y_pred_test)*100
	
    results = compute_metrics(
                    prediction=y_pred_test, #pred[data.test_mask].detach().cpu().numpy(),
                    target=test_labels, #data.y[data.test_mask].detach().cpu().numpy(),
                    n_classes=int(test_labels.max())
    )
	
    aa_acc, oa_acc, kappa, aa_acc_all = results['AA'], results['OA'], results['Kappa'], results['PA']#'''
	
    print (' [The verification accuracy is: %.2f and %.2f] ' %(acc, aa_acc) )
    return acc

def test(model, device, test_loader, args, writer, epoch, num_result, result_writer, PATH_result, load_checkpoints=False):
    model.eval()
    with torch.no_grad():
        results = test_main(model, device, test_loader, args, writer, epoch, num_result, result_writer, PATH_result, load_checkpoints=False) 
 
    return results

def main(Train_Contiune_Mode = True, Only_Test_Mode = False):

    if args.dataset == 'Berlin':
        args.hsi_bands = 10
        args.sar_bands = 4
        args.num_class = 8
    elif args.dataset == 'Augsburg':
        args.hsi_bands = 30##180
        args.sar_bands = 4
        args.num_class = 7
    elif args.dataset == 'HHK':
        args.hsi_bands = 166
        args.sar_bands = 3
        args.num_class = 5
    elif args.dataset == 'Muufl':
        args.hsi_bands = 30##64
        args.sar_bands = 2
        args.num_class = 11
    elif args.dataset == 'Houston':
        args.hsi_bands = 30 #144
        args.sar_bands = 1
        args.num_class = 15
    elif args.dataset == 'Trento':
        args.hsi_bands = 30 ##63
        args.sar_bands = 2
        args.num_class = 6
        
        
    train_loader, valid_loader = data_loader_train.build_datasets(args)
    labels_text = data_loader_train.get_dataset_info(args.dataset)
    test_loader = data_loader_test.data_loader(args)
    

    PATH_result = args.result_path + args.dataset 
    ensure_dir(PATH_result+"/")
    PATH_model = args.checkpoint_path + args.dataset 
    ensure_dir(PATH_model+"/")

    for num_result in range (3): 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

        config = config_2Dmf
        model = GCCQTNet(config, device, args.hsi_bands, args.sar_bands, args.patch_size, args.num_class).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
        
        Net_params, slim_params = [], []
        i=0
        for name, param in model.named_parameters():
            if param.requires_grad and name.endswith('weight') and 'bn2' in name:
                if len(slim_params) % 2 == 0:
                    slim_params.append(param[:len(param) // 2])
                else:
                    slim_params.append(param[len(param) // 2:])
                
        best_acc = 0
        statistics = []
        end = time()
        model_path = PATH_model +"{}".format(num_result)+ '/' + 'model.pth'
        rep = 1
        writer = SummaryWriter(log_dir = PATH_model +"{}".format(num_result)) 
        epochs_ini = 0
        if Only_Test_Mode is True and model_path is not None:
            print("Loading weights from trained deep model to continue testing only....")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint)
            model.eval()
            result_writer = write2csv(args, PATH_result, num_result = num_result)
            with torch.no_grad():
                results = test(model, device, test_loader, args, writer, epoch, num_result, result_writer, PATH_result, load_checkpoints=True)
            statistics.append(results)
            cm_viz(writer, cm=results["Confusion matrix"], labels_text=labels_text, replica=rep)
            metrics2text(results=results, labels_text=labels_text, replica=rep)#''' 
            print('End of testing!!!')
        else:
            if Train_Contiune_Mode==True and model_path is not None:
                epochs_ini = 1150
                model_path = PATH_model +"{}".format(num_result)+ '/' + 'test_model_{}.pth'.format(epochs_ini)
                print("Loading weights from trained deep model to continue training....")
                print(model_path)
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint)
            else:
                # initialize_weights(model)
                print("Training from scratch and testing....")
                if not os.path.exists(PATH_model+"{}".format(num_result)+'/'): os.makedirs(PATH_model+"{}".format(num_result)+'/')
            if not os.path.exists(PATH_result+"{}".format(num_result)+'/'): os.makedirs(PATH_result+"{}".format(num_result)+'/')
            result_writer = write2csv(args, PATH_result, num_result = num_result)
            for epoch in range(epochs_ini, args.epochs):
                train(model, device, train_loader, args.batch_size, optimizer, writer, epoch, slim_params, args.bn_threshold) 
                
                if (epoch+1)%100 == 0:
                    results = test(model, device, test_loader, args, writer, epoch, num_result, result_writer, PATH_result, load_checkpoints=False)
                    cm_viz(writer, cm=results["Confusion matrix"], labels_text=labels_text, replica=rep)
                    model_path = PATH_model +"{}".format(num_result)+ '/' + 'test_model_{}.pth'.format(epoch+1)
                    torch.save(model.state_dict(), model_path)
        end2 = time()
        minutes = int((end2 - end) / 60)
        seconds = int((end2 - end) - minutes * 60)
        print("Training Finished!!! Training time：", minutes, "m", seconds, "s")
        
        result_writer.close()
        writer.close()#
if __name__ == '__main__':
    main(Train_Contiune_Mode = False, Only_Test_Mode = False)
