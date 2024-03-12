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

from model import UNet3D

np.random.seed(555)
torch.manual_seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--m', '-m', default=0, type=int, help= \
    '1: train, 2: test')
parser.add_argument('--gpu', '-g', default=0, type=int)
parser.add_argument('--exp', '-exp', type=str, help='exp name')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'dataset_processed'
save_dir = f'results/{args.exp}'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

x_train = np.load(data_dir+'train_total.npy')
y_train = np.load(data_dir+'train_label.npy')

x_val = np.load(data_dir+'val_total.npy')
y_val = np.load(data_dir+'val_label.npy')

x_test = np.load(data_dir+'test_total.npy')
y_test = np.load(data_dir+'test_label.npy')

class CustomDataset(Dataset):
    def __init__(self, img, Y):
        self.img = torch.Tensor(img).float()
        self.Y = torch.Tensor(Y).int()
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        
        img_idx = self.img[idx]
        Y_idx = self.Y[idx]
    
        return img_idx, Y_idx

train_dataset = CustomDataset(x_train, y_train)
val_dataset = CustomDataset(x_val, y_val)
test_dataset = CustomDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset), shuffle=True) #For using FindBoundary, we have to set BATCH_SIZE as BATCH_SIZE


model = UNet3D(in_channels=IN_CHANNELS, num_classes = 2).to(device)

if args.m == 1:
    dataloader = train_dataloader 
elif args.m == 2:
    dataloader = test_dataloader
    model = torch.load(save_dir+'/model.pth', map_location=device)
    model_state_dict = torch.load(save_dir+'/model_state_dict.pth')['model_state_dict']
    model.load_state_dict(model_state_dict)

criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch, last_epoch=-1, verbose=False)


total_loss = []
best_loss = 1000

# Training loop
for epoch in range(100):
    print('epoch: ', epoch)
    with tqdm(dataloader, unit='batch') as tepoch:
        batch_loss = 0
        for idx, (img, y) in enumerate(tepoch):
            optimizer.zero_grad()
            
            batch_size = img.shape[0]
            pred = model(img)
            loss = criterion(pred, y)
            tepoch.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

            batch_loss += loss.item() / batch_size

            if args.m == 2:
                
                fig, axes = plt.subplots(1, 1, figsize=(12, 12))
                axes = axes.ravel()

                fpr, tpr, thresholds = roc_curve(y, pred)
                roc_auc = auc(fpr, tpr)
                    
                axes[0].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
                axes[0].plot([0, 1], [0, 1], 'k--')
                axes[0].set_xlim([0.0, 1.0])
                axes[0].set_ylim([0.0, 1.05])
                axes[0].set_xlabel('False Positive Rate')
                axes[0].set_ylabel('True Positive Rate')
                axes[0].set_title(f'ROC Curve for ONJ classification')
                axes[0].legend(loc='lower right')
                
                plt.tight_layout()
                plt.savefig(save_dir+f'/test.jpg')
                plt.close()

                
    total_loss.append(batch_loss.item())   
    
    if batch_loss.item() < best_loss:

        torch.save({
                    'model_state_dict': model.state_dict(),
                    'loss': batch_loss},
                    save_dir+'/'+'model.pth')

        torch.save(model, save_dir+'/'+'model.pth')

        best_loss = batch_loss.item()
                                
        else:
            pass
        
            
    print('loss: ', batch_loss.item())
        

    total_loss = np.array(total_loss)

    plt.plot(total_loss, label='train')
    # plt.plot(total_loss_val, label='val')
    plt.legend()
    plt.savefig(save_dir+'/train_loss.jpg')
    plt.close()





