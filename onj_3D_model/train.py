import os
import sys

sys.path.append(os.getcwd())

from data_func import load_data_dict, patient_dicts
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import cv2
import numpy as np
from enum import Enum
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from model import UNet3D
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch.distributed as dist

np.random.seed(555)
torch.manual_seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--m', '-m', default=0, type=int, help= \
    '1: train, 2: test')
parser.add_argument('--gpu', '-g', default=0, type=int)
parser.add_argument('--exp', '-exp', type=str, help='exp name')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'dataset_processed'
save_dir = f'results/{args.exp}'
BATCH_SIZE = 1
IN_CHANNELS = 1 # channel 
BCE_WEIGHTS = [0.004, 0.996]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# x_train = np.load(data_dir+'/'+'train_total_mini.npy') # memory
# y_train = np.load(data_dir+'/'+'train_label_mini.npy')



x_val = np.load(data_dir+'/'+'val_total.npy')
# y_val = np.load(data_dir+'/'+'val_label.npy')

# x_test = np.load(data_dir+'/'+'test_total.npy')
# y_test = np.load(data_dir+'/'+'test_label.npy')


class CustomDataset(Dataset):
    def __init__(self, split):
        if split == 'train':

            self.img = np.load(data_dir+f'/{split}_total.npy', mmap_mode='r')
            self.Y = np.load(data_dir+f'/{split}_label.npy', mmap_mode='r')
        elif split in ['test', 'val']:
            
            self.img = np.load(data_dir+f'/{split}_total.npy', mmap_mode='r')
            self.Y = np.load(data_dir+f'/{split}_label.npy', mmap_mode='r')
            if split == 'val':
                self.img = self.img[:10]
                self.Y = self.Y[:10]


        # self.img = torch.Tensor(self.img).float().to(device)
        # self.Y = torch.Tensor(self.Y).float().to(device)
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img_idx = self.load_data(self.img, idx)
        img_idx = np.asarray(img_idx).astype(np.float32)
        Y_idx = self.load_data(self.Y, idx)
        Y_idx = np.asarray(Y_idx).astype(np.float32)
        
        # img_tensor = torch.from_numpy(img_idx).float().to(device)
        # Y_tensor = torch.from_numpy(Y_idx).float().to(device)
    
        return img_idx, Y_idx

    def load_data(self, mmap_array, idx):
        return mmap_array[idx]



### distributed data parallel

dist.init_process_group("nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

dist.destroy_process_group()

dist.init_process_group(backend='nccl')
model = UNet3D(in_channels=IN_CHANNELS, num_classes = 1).cuda(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


if args.m == 1:
    train_dataset = CustomDataset('train')
    val_dataset = CustomDataset('val')
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True)

    dataloader = train_dataloader 
    model.train()
    EPOCH = 30
elif args.m == 2:
    test_dataset = CustomDataset('test')
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=True) #For using FindBoundary, we have to set BATCH_SIZE as BATCH_SIZE



    dataloader = test_dataloader
    model = torch.load(save_dir+'/model.pth', map_location=device)
    model_state_dict = torch.load(save_dir+'/model_state_dict.pth')['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()
    EPOCH = 1

# criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
criterion = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch, last_epoch=-1, verbose=False)


total_loss = []
total_val_loss = []
best_loss = 1000

# Training loop

for epoch in range(EPOCH):
    print('epoch: ', epoch)
    y_list = []
    pred_list = []
    
    with tqdm(dataloader, unit='batch') as tepoch:
        batch_loss = 0
        for idx, (img, y) in enumerate(tepoch):
            optimizer.zero_grad()

            img = img.to(device)
            y = y.to(device)
            
            # img = (BS, 70, 512, 512)
            batch_size = img.shape[0]
            img = img.unsqueeze(1)
            y = y.unsqueeze(1)
            
            
            pred, _ = model(img)

            loss = criterion(pred, y)
            tepoch.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

            if args.m == 2:
                y_list = np.append(y_list, y.cpu().detach().numpy())
                pred_list = np.append(pred_list, pred.cpu().detach().numpy())

            batch_loss += loss.item() / batch_size



        batch_loss /= idx
        total_loss = np.append(total_loss, batch_loss)
    if args.m == 1:       
        ## check validation loss

        with torch.no_grad():

            val_batch_loss = 0
            for val_idx, (val_img, val_y) in enumerate(val_dataloader):

                batch_size = img.shape[0]
                img = img.unsqueeze(1)
                y = y.unsqueeze(1)
            
                
                val_batch_size = val_img.shape[0]
                val_img = val_img.unsqueeze(1)
                val_y = val_y.unsqueeze(1)
                val_pred = model(val_img)
                val_loss = criterion(val_pred, val_y)
                val_batch_loss += val_loss.item() / val_batch_size

            val_batch_loss /= val_idx

            total_val_loss = np.append(total_val_loss, val_batch_loss)

            if val_batch_loss < best_loss:

                torch.save({
                            'model_state_dict': model.state_dict(),
                            'loss': batch_loss},
                            save_dir+'/'+'model_state_dict.pth')

                torch.save(model, save_dir+'/'+'model.pth')

                best_loss = val_batch_loss
                                        
            else:
                pass
        
            
        print('train loss: ', batch_loss, 'val loss: ', val_batch_loss)
            

        total_loss = np.array(total_loss)
        total_val_loss = np.array(total_val_loss)

        plt.plot(total_loss, label='train')
        plt.plot(total_val_loss, label='val')
        plt.legend()
        plt.savefig(save_dir+'/train_loss.jpg')
        plt.close()


if args.m == 2:
    print(y_list.shape, pred_list.shape)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    fpr, tpr, thresholds = roc_curve(y_list, pred_list)
    roc_auc = auc(fpr, tpr)
        
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for ONJ classification')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_dir+f'/test.jpg')
    plt.close()



