# Import stuff
import argparse
import os
import random
import logging
import numpy as np
import pandas as pd
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim

import torch.distributed as dist
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import nibabel as nib

from Models.TABS_Model import TABS
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

# Root directory
parser.add_argument('--root', default='./', type=str)

# learning rate
parser.add_argument('--lr', default=0.00001, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=300, type=int)

parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--gpu_available', default='0', type=str)

args = parser.parse_args()

def main_worker():
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.device(args.gpu)  # discouraged

    # 모델을 선언
    model = TABS()
    # GPU에 해당 모델을 올린다.
    model.cuda(args.gpu)
    print('Model Built!')

    # Using adam optimizer (amsgrad variant) with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # MSE loss for this task (regression). Using reduction value of sum because we want to specify the number of voxels to divide by (only in the brain map)
    # Loss function을 정의
    criterion = nn.MSELoss(reduction='sum')
    criterion = criterion.cuda(args.gpu)

    # *************************************************************************
    # Place train and validation datasets/dataloaders here
    # *************************************************************************
    # Dataset
    class MRI_Datasets(Dataset):
        def __init__(self, file_list, transform=None, target_transform=None):
            self.file_list = file_list # train이면 train, val이면 val, test면 test
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.file_list['image'])

        def __getitem__(self, idx):

            img_path = self.file_list.loc[idx, 'image']
            image = nib.load(img_path)
            image = image.get_fdata()
            image = torch.from_numpy(image).float()


            gm_label_path = self.file_list.loc[idx, 'gm_label']
            gm_label = nib.load(gm_label_path)
            gm_label = gm_label.get_fdata()
            gm_label = torch.from_numpy(gm_label).float()


            wm_label_path = self.file_list.loc[idx, 'wm_label']
            wm_label = nib.load(wm_label_path)
            wm_label = wm_label.get_fdata()
            wm_label = torch.from_numpy(wm_label).float()


            csf_label_path = self.file_list.loc[idx, 'csf_label']
            csf_label = nib.load(csf_label_path)
            csf_label = csf_label.get_fdata()
            csf_label = torch.from_numpy(csf_label).float()

            label = torch.stack([gm_label, wm_label, csf_label], dim=0)



            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

    train_list = pd.read_csv('./train_file.csv', index_col=0)
    val_list = pd.read_csv('./val_file.csv', index_col=0)
    test_list = pd.read_csv('./test_file.csv', index_col=0)

    train_set = MRI_Datasets(file_list=train_list)
    val_set = MRI_Datasets(file_list=val_list)
    test_set = MRI_Datasets(file_list=test_list)

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)


    start_time = time.time()

    # Enable gradient calculation for training
    torch.set_grad_enabled(True)

    # Declare lists to keep track of training and val losses over the epochs
    # Epoch 마다 train loss와 val loss를 기록하기 위한 list를 선언
    train_global_losses = []
    val_global_losses = []
    best_epoch = 0

    print('Start to train!')

    # Main training/validation loop
    # 설정한 epoch만큼 돈다.
    for epoch in range(args.start_epoch, args.end_epoch):

        # Declare lists to keep track of losses and metrics within the epoch
        train_epoch_losses = []
        val_epoch_losses = []
        val_epoch_pcorr = []
        val_epoch_psnr = []
        start_epoch = time.time()
        # Start Training
        model.train()

        # Loop through train dataloader here. OK
        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            mri_images, targets = data
            mri_images = mri_images.view(-1, 1, 192,192,192).float()
            targets = targets.view(-1, 3, 192, 192, 192).float()

            # 이미지와 타겟을 gpu에 올리는 과정
            mri_images = mri_images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            # 실질적으로 모델을 학습하고 Loss 값을 구하는 과정
            loss, isolated_images, stacked_brain_map = get_loss(model, criterion, mri_images, targets, 'train')

            train_epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Transition to val mode
        model.eval()

        with torch.no_grad():

            # Loop through validation dataloader here. OK
            for i, data in enumerate(val_loader):
                # Sample data for the purpose of demonstration
                mri_images, targets = data

                mri_images = mri_images.view(-1, 1, 192, 192, 192).float()
                targets = targets.view(-1, 3, 192, 192, 192).float()

                mri_images = mri_images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

                loss, isolated_images, stacked_brain_map = get_loss(model, criterion, mri_images, targets, 'val')

                val_epoch_losses.append(loss.item())

                for j in range(0,len(isolated_images)):
                    cur_pcorr = overall_metrics(isolated_images[j], targets[j], stacked_brain_map[j])
                    val_epoch_pcorr.append(cur_pcorr)

        end_epoch = time.time()

        # Average train and val loss over every MRI scan in the epoch. Save to global losses which tracks across epochs
        train_net_loss = sum(train_epoch_losses) / len(train_epoch_losses)
        val_net_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        train_global_losses.append(train_net_loss)
        val_global_losses.append(val_net_loss)
        pcorr = sum(val_epoch_pcorr) / len(val_epoch_pcorr)

        print('Epoch: {} | Train Loss: {:.4f} | Val Loss: {:.4f} | Pearson: {:.4f}'.format(epoch, train_net_loss, val_net_loss, pcorr))

        checkpoint_dir = args.root
        # Save the model if it reaches a new min validation loss
        if val_global_losses[-1] == min(val_global_losses):

            # Only save model at higher epochs
            if epoch > 200:
                print('saving model at the end of epoch ' + str(epoch))
                best_epoch = epoch
                file_name = os.path.join(checkpoint_dir,
                                         'TABS_model_epoch_{}_val_loss_{:.4f}.pth'.format(epoch, val_global_losses[-1]))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    },
                    file_name)

    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    print('The total training time is {:.2f} hours'.format(total_time))

    print('----------------------------------The training process finished!-----------------------------------')

    # log_name = os.path.join(args.root, args.protocol, 'loss_log_restransunet.txt')
    log_name = os.path.join(args.root, 'loss_log_TABS.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Loss (%s) ================\n' % now)
        log_file.write('best_epoch: ' + str(best_epoch) + '\n\n')
        log_file.write('train_losses: ')
        log_file.write('%s\n\n' % train_global_losses)
        log_file.write('val_losses: ')
        log_file.write('%s\n\n' % val_global_losses)
        log_file.write('train_time: ' + str(total_time) + '\n\n')

    learning_curve(best_epoch, train_global_losses, val_global_losses)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Input the best epoch, lists of global (across epochs) train and val losses. Plot learning curve
def learning_curve(best_epoch, train_global_losses, val_global_losses):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Epochs')
    ax1.set_xticks(np.arange(0, int(len(train_global_losses) + 1), 10))

    ax1.set_ylabel('Loss')
    ax1.plot(train_global_losses, '-r', label='Training loss', markersize=3)
    ax1.plot(val_global_losses, '-b', label='Validation loss', markersize=3)
    ax1.axvline(best_epoch, color='m', lw=4, alpha=0.5, label='Best epoch')
    ax1.legend(loc='upper left')
    save_name = 'Learning_Curve_TABS' + '.png'
    plt.savefig(os.path.join(args.root, save_name))

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch / max_epoch), power), 8)

# Calculate pearson correlation and psnr only between the voxels of the brain map (do by total brain not tissue type during training)
def overall_metrics(isolated_image, target, stacked_brain_map):
    # Flatten the GT, isolated output, and brain mask
    GT_flattened = torch.flatten(target)
    iso_flattened = torch.flatten(isolated_image)
    mask_flattened = torch.flatten(stacked_brain_map)

    # Only save the part of the flattened GT/output that corresponds to nonzero values of the brain mask
    GT_flattened = GT_flattened[mask_flattened.nonzero(as_tuple=True)]
    iso_flattened = iso_flattened[mask_flattened.nonzero(as_tuple=True)]

    iso_flattened = iso_flattened.cpu().detach().numpy()
    GT_flattened = GT_flattened.cpu().detach().numpy()

    pearson = np.corrcoef(iso_flattened, GT_flattened)[0][1]

    return pearson

# Given the model, criterion, input, and GT,
# this function calculates the loss and returns the isolated output (stripped of background) and brain map
# 모델, criterion(손실 함수), input, Ground Truth가 주어졌을 때(GT가 하나인지 3개인지 명확하게 파악해야 한다!)
# 이 함수는 loss를 계산하고, isolated output을 반환한다.

def get_loss(model, criterion, mri_images, targets, mode):
    # mode가 validation인 경우
    if mode == 'val':
        # seed 지정
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Gen model outputs
    # input 값을 넣어서 output을 생성하는 과정
    output = model(mri_images)

    # Construct binary brain map to consider loss only within there
    input_squeezed = torch.squeeze(mri_images,dim=1)
    brain_map = (input_squeezed > -1).float()
    stacked_brain_map = torch.stack([brain_map, brain_map, brain_map], dim=1)

    # Zero out the background of the segmentation output
    isolated_images = torch.mul(stacked_brain_map, output)

    # Calculate loss over just the brain map
    loss = criterion(isolated_images, targets)
    num_brain_voxels = stacked_brain_map.sum()
    loss = loss / num_brain_voxels

    return loss, isolated_images, stacked_brain_map

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_available
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
