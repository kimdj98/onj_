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

import SimpleITK as sitk


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

data_dir = 'dataset_processed_512' ## this data is unnormalized data
save_dir = f'results/{args.exp}'
BATCH_SIZE = 1
ACCUMULATION_STEPS=4
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
def adjust_intensity_numpy(image_array, factor=1.5):
    """Adjust the intensity of the 3D numpy array directly."""
    # Assuming image_array is a numpy array and already normalized (e.g., 0 to 1 range)
    # Adjust the intensity
    adjusted_image_array = image_array * factor
    
    # Clip the values to maintain the expected range, if necessary
    adjusted_image_array = np.clip(adjusted_image_array, 0, 1)
    
    return adjusted_image_array

def add_gaussian_noise_3d(image, mean=0.0, std=0.1):
    """Add Gaussian noise to the 3D image."""
    noise_filter = sitk.AdditiveGaussianNoiseImageFilter()
    noise_filter.SetMean(mean)
    noise_filter.SetStandardDeviation(std)
    return noise_filter.Execute(image)

def rotate_img_3d(image, angle_degrees, rotation_axis):
    """
    Rotate a 3D image around a specified axis.
    
    Parameters:
    - image: The SimpleITK image to rotate.
    - angle_degrees: The rotation angle in degrees.
    - rotation_axis: The axis of rotation (e.g., (1, 0, 0) for x-axis).
    
    Returns:
    - The rotated SimpleITK image.
    """
    # Convert angle from degrees to radians
    angle_radians = np.deg2rad(angle_degrees)
    
    # Get the image center
    image_center = image.TransformContinuousIndexToPhysicalPoint([(index - 1) / 2.0 for index in image.GetSize()])
    
    # Create a 3D Euler transformation
    transform = sitk.Euler3DTransform(image_center, rotation_axis[0], rotation_axis[1], rotation_axis[2], angle_radians)
    
    # Resample the image using the transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    rotated_image = resampler.Execute(image)
    
    return rotated_image

class CustomDataset(Dataset):
    def __init__(self, split):
        if split == 'train':

            self.img = np.load(data_dir+f'/{split}_total.npy', mmap_mode='r')
            self.Y = np.load(data_dir+f'/{split}_label.npy', mmap_mode='r')
            self.seg = np.load(data_dir+f'/{split}_seg.npy', mmap_mode='r')

            ### original slices
            # self.img = np.load(data_dir+f'/{split}_total_org.npz', mmap_mode='r')
            # self.Y = np.load(data_dir+f'/{split}_label.npy', mmap_mode='r')
            # self.seg = np.load(data_dir+f'/{split}_seg.npy', mmap_mode='r')

        elif split in ['test', 'val']:
            
            self.img = np.load(data_dir+f'/{split}_total.npy', mmap_mode='r')
            self.Y = np.load(data_dir+f'/{split}_label.npy', mmap_mode='r')
            self.seg = np.load(data_dir+f'/{split}_seg.npy', mmap_mode='r')

            # self.img = np.load(data_dir+f'/{split}_total_org.npz', mmap_mode='r')
            # self.Y = np.load(data_dir+f'/{split}_label.npy', mmap_mode='r')
            # self.seg = np.load(data_dir+f'/{split}_seg.npy', mmap_mode='r')

            if split == 'val':
                self.img = self.img[:10]
                self.Y = self.Y[:10]
                self.seg = self.seg[:10]



        lb = np.percentile(self.img, 1)
        ub = np.percentile(self.img, 99)
        self.img = np.clip(self.img, lb, ub)

        ## standardization (z-score normalization)
        self.mean = np.mean(self.img)
        self.std = np.std(self.img)
        self.img = (self.img - self.mean) / self.std



        ## robust scaling
        # median = np.median(self.img)
        # q1 = np.percentile(self.img, 25)
        # q3 = np.percentile(self.img, 75)
        # iqr = q3 - q1
        # self.img = (self.img - median) / iqr

        ## log transformation
        # self.img = self.img + 1e-6
        # self.img = np.log(self.img)

        ## ! How should we deal with segmentation mask with augmented images?
        # if split == 'train':
            # aug_imgs, aug_Y = self.augment_data(self.img, self.Y)
            # self.img = np.concatenate((self.img, aug_imgs), axis=0)
            # self.Y = np.concatenate((self.Y, aug_Y), axis=0)


    def augment_data(self, images, labels):
        # This method performs augmentation on the input images and labels
        # For simplicity, let's use adding Gaussian noise as an example of augmentation
        augmented_images = []
        for image in images:
            sitk_image = sitk.GetImageFromArray(image)
            noisy_image = self.add_gaussian_noise_3d(sitk_image)
            noisy_image_array = sitk.GetArrayFromImage(noisy_image)
            augmented_images.append(noisy_image_array)
        
        augmented_images = np.array(augmented_images)
        
        # For labels, no augmentation is needed, just copy
        augmented_labels = np.copy(labels)
        
        return augmented_images, augmented_labels 

    def add_gaussian_noise_3d(self, image):
        noise_filter = sitk.AdditiveGaussianNoiseImageFilter()
        noise_filter.SetMean(0.0)
        noise_filter.SetStandardDeviation(0.1)
        noisy_image = noise_filter.Execute(image)
        return noisy_image
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img_idx = self.load_data(self.img, idx)
        img_idx = np.asarray(img_idx).astype(np.float32)
        Y_idx = self.load_data(self.Y, idx)
        Y_idx = np.asarray(Y_idx).astype(np.float32)

        seg_idx = self.load_data(self.seg, idx)
        seg_idx = np.asarray(seg_idx).astype(np.float32)


        # current_depth = img_idx.shape[0]
        # if current_depth < 64:
        #     # Padding
        #     pad_size = 64 - current_depth
        #     pad_before = pad_size // 2
        #     pad_after = pad_size - pad_before
        #     img_idx_padded = np.pad(img_idx, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
        # elif current_depth > 64:
        #     # Cropping or Resampling (Here we'll just crop for simplicity)
        #     start_slice = (current_depth - 64) // 2
        #     img_idx_padded = img_idx[start_slice:start_slice + 64, :, :]
        # else:
        #     img_idx_padded = img_idx

    
        return img_idx, Y_idx, seg_idx, self.mean, self.std
        # print(img_idx.shape)
        # quit()
        # return img_idx, Y_idx

    def load_data(self, mmap_array, idx):
        return mmap_array[idx]
        # return mmap_array[f'{idx}']



### distributed data parallel

dist.init_process_group("nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

dist.destroy_process_group()

dist.init_process_group(backend='nccl')
model = UNet3D(in_channels=IN_CHANNELS, num_classes = 1).cuda(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)


if args.m == 1:
    train_dataset = CustomDataset('train')
    val_dataset = CustomDataset('val')
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True)

    dataloader = train_dataloader 
    model.train()
    EPOCH = 100
elif args.m == 2:
    test_dataset = CustomDataset('test')
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=True) #For using FindBoundary, we have to set BATCH_SIZE as BATCH_SIZE

    ## FOR CHECKING TRAIN SET
    # test_dataset = CustomDataset('train')
    # test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=True)


    dataloader = test_dataloader
    model = torch.load(save_dir+'/model.pth', map_location=device)
    model_state_dict = torch.load(save_dir+'/model_state_dict.pth')['model_state_dict']
    model.load_state_dict(model_state_dict)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    model = model.cuda(local_rank)

    model.eval()
    EPOCH = 1

# criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
criterion = torch.nn.BCELoss()
criterion_seg = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor([0.2]).to(device))
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
        for idx, (img, y, seg, mean, std) in enumerate(tepoch):
            optimizer.zero_grad()


            img = img.to(device)
            y = y.to(device)
            seg = seg.to(device)
            
            # img = (BS, 64, 256, 256)
            batch_size = img.shape[0]
            img = img.unsqueeze(1)
            y = y.unsqueeze(1)
            seg = seg.unsqueeze(1)

            pred, pred_seg = model(img)

            loss_cls = criterion(pred, y) / ACCUMULATION_STEPS
            # print(pred_seg.shape, seg.shape)
            loss_seg = criterion_seg(pred_seg, seg.float())

            
            loss = loss_cls + loss_seg
            # print(loss_cls, loss_seg)
            tepoch.set_postfix(loss=loss.item())
            loss.backward()

            if (idx + 1) % ACCUMULATION_STEPS == 0 or (idx + 1) == len(dataloader):
                optimizer.step()  # Perform an optimization step
                model.zero_grad()  # Clear the gradients


            if args.m == 2:
                # pred = torch.sigmoid(pred)
                # pred = (pred>=0.5).long()

                ## BATCHSIZE 1
                slices_GT = img.squeeze(1).cpu().detach().numpy() #(bs, 64, 256, 256)
                seg_pred = pred_seg.squeeze(1).cpu().detach().numpy()
                seg_GT = seg.squeeze(1).cpu().detach().numpy()

                mean = mean.cpu().detach().numpy()
                std = std.cpu().detach().numpy()

                for slice_idx in range(slices_GT.shape[1]):
                    
                    if slice_idx in range(int(slices_GT.shape[1]*0.4), int(slices_GT.shape[1]*0.6)):
                        seg_GT_plot = seg_GT[0, slice_idx, :,:]
                        seg_pred_plot = seg_pred[0, slice_idx, :, :]

                        img_GT = slices_GT[0, slice_idx, :, :]


                        img_GT = img_GT * std + mean

                        if 1 in seg_GT_plot:
                            rows = np.any(seg_GT_plot, axis=1)
                            cols = np.any(seg_GT_plot, axis=0)
                            row_min, row_max = np.where(rows)[0][[0, -1]]
                            col_min, col_max = np.where(cols)[0][[0, -1]]
                            bbox = patches.Rectangle((col_min, row_min), col_max-col_min, row_max-row_min,
                                                linewidth=2, edgecolor='r', facecolor='none')

                            fig, axes = plt.subplots(1, 2)
                            axes[0].imshow(img_GT, cmap='gray')
                            axes[0].add_patch(bbox)
                            axes[0].title('Original')

                            axes[1].imshow(img_GT, cmap='gray')
                            axes[1].imshow(seg_GT_plot, alpha=0.2)
                            axes[1].title('Prediction')
                            plt.savefig(save_dir+f'/seg_{idx}_{slice_idx}.jpg')
                            plt.close()


                y_list = np.append(y_list, y.cpu().detach().numpy())
                pred_list = np.append(pred_list, pred.cpu().detach().numpy())

            batch_loss += loss.item() / batch_size



        batch_loss /= idx
        total_loss = np.append(total_loss, batch_loss)
    if args.m == 1:       
        ## check validation loss

        with torch.no_grad():

            val_batch_loss = 0
            for val_idx, (val_img, val_y, val_seg) in enumerate(val_dataloader):
                val_img = val_img.to(device)
                val_y = val_y.to(device)
                val_seg = val_seg.to(device)
            
                
                val_batch_size = val_img.shape[0]
                val_img = val_img.unsqueeze(1)
                val_y = val_y.unsqueeze(1)
                val_seg = val_seg.unsqueeze(1)
                val_pred, val_pred_seg = model(val_img)
                val_loss_cls = criterion(val_pred, val_y)
                val_loss_seg = criterion_seg(val_pred_seg, val_seg)
                val_loss = val_loss_cls + val_loss_seg
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
    print(y_list, pred_list)

    
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



