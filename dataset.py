import os
import cv2
import math
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VimeoDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448
        self.data_root = 'D:\\dataset\\vimeo_septuplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(math.floor(len(self.trainlist) * 0.95))
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        # RIFEm with Vimeo-triplet
        '''        
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        '''
    
        # RIFEm with Vimeo-septuplet
        imgpaths = [
            imgpath + '/im1.png', 
            imgpath + '/im2.png', 
            imgpath + '/im3.png', 
            imgpath + '/im4.png', 
            imgpath + '/im5.png', 
            imgpath + '/im6.png', 
            imgpath + '/im7.png'
            ]

        ind = [0, 1, 2, 3, 4, 5, 6]
        random.shuffle(ind)
        ind = ind[:3]
        ind.sort()
        img0 = cv2.imread(imgpaths[ind[0]], cv2.IMREAD_COLOR)
        gt   = cv2.imread(imgpaths[ind[1]], cv2.IMREAD_COLOR)
        img1 = cv2.imread(imgpaths[ind[2]], cv2.IMREAD_COLOR)
        timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        return img0, gt, img1, timestep

    def check(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [
            imgpath + '/im1.png', 
            imgpath + '/im2.png', 
            imgpath + '/im3.png', 
            imgpath + '/im4.png', 
            imgpath + '/im5.png', 
            imgpath + '/im6.png', 
            imgpath + '/im7.png'
            ]
            
        ind = [0, 1, 2, 3, 4, 5, 6]
        for i in ind:
            img = cv2.imread(imgpaths[ind[i]], cv2.IMREAD_COLOR)
            if(img.shape[0] != 256 or img.shape[1] != 448 or img.shape[2]!= 3):
                print(imgpaths[ind[i]], img.shape)
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.crop(img0, gt, img1, 256, 256)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep
    
class VimeoDatasetSFT(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name        
        self.data_root = 'D:\\dataset\\vimeo_ex'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'train.txt')
        test_fn = os.path.join(self.data_root, 'test.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(math.floor(len(self.trainlist) * 0.95))
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [
            imgpath + '/im1.png', 
            imgpath + '/im2.png', 
            imgpath + '/im3.png', 
            imgpath + '/im4.png', 
            imgpath + '/im5.png', 
            imgpath + '/im6.png', 
            imgpath + '/im7.png'
            ]

        ind = [0, 1, 2, 3, 4, 5, 6]
        random.shuffle(ind)
        ind = ind[:3]
        ind.sort()
        img0 = cv2.imread(imgpaths[ind[0]], cv2.IMREAD_COLOR)
        gt   = cv2.imread(imgpaths[ind[1]], cv2.IMREAD_COLOR)
        img1 = cv2.imread(imgpaths[ind[2]], cv2.IMREAD_COLOR)
        timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        return img0, gt, img1, timestep

    def check(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [
            imgpath + '/im1.png', 
            imgpath + '/im2.png', 
            imgpath + '/im3.png', 
            imgpath + '/im4.png', 
            imgpath + '/im5.png', 
            imgpath + '/im6.png', 
            imgpath + '/im7.png'
            ]
            
        ind = [0, 1, 2, 3, 4, 5, 6]
        for i in ind:
            img = cv2.imread(imgpaths[ind[i]], cv2.IMREAD_COLOR)
            if(img.shape[0] != 1080 or img.shape[1] != 1920 or img.shape[2]!= 3):
                print(imgpaths[ind[i]], img.shape)
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        img0, gt, img1 = self.crop(img0, gt, img1, 1024, 1920)
        if self.dataset_name == 'train':
            if random.uniform(0, 1) < 0.5:
                # flip channel
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt   = gt[:, :, ::-1]

            if random.uniform(0, 1) < 0.5:
                # flip horizon
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt   = gt[:, ::-1]
            
            if random.uniform(0, 1) < 0.5:
                # flip time
                tmp  = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt   = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep

if __name__ == "__main__":
    dataset_tra = VimeoDatasetSFT('train')
    dataset_val = VimeoDatasetSFT('val')

    for i in tqdm(range(0, len(dataset_tra)), desc="checking train"):
        dataset_tra.check(i)

    for i in tqdm(range(0, len(dataset_val)), desc="checking val"):
        dataset_val.check(i)