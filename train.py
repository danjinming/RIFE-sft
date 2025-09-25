import os
import math
import torch
import numpy as np
import random
import argparse
import lpips

from rife import Model
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda")
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

log_path = 'runs'
writer = SummaryWriter(log_path + '/train')
writer_val = SummaryWriter(log_path + '/validate')

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return 2e-4 * mul + 1e-7

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model):
    dataset = VimeoDataset('train')
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, shuffle=True)
    
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=8, num_workers=8, pin_memory=True, shuffle=False)

    step = 0
    args.step_per_epoch = train_data.__len__()
    print('training...')
    
    for epoch in range(1, args.epoch + 1):
        for i, data in enumerate(train_data):
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:,  :6]
            gt   = data_gpu[:, 6:9]

            # flip and cat to extend double batch            
            imgs = torch.cat((imgs, imgs.flip(-1)), 0)
            gt   = torch.cat((gt, gt.flip(-1)), 0)
            timestep = torch.cat((timestep, timestep.flip(-1)), 0)

            lr = get_learning_rate(step)
            pred, info = model.update(imgs, gt, lr, training=True, timestep=timestep)

            if step % 200 == 1:
                writer.add_scalar('lr', lr, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/cons', info['loss_cons'], step)
                writer.add_scalar('loss/time', info['loss_time'], step)
                writer.add_scalar('loss/encode', info['loss_encode'], step)
                writer.add_scalar('loss/vgg', info['loss_vgg'], step)
                writer.add_scalar('loss/gram', info['loss_gram'], step)
                writer.flush()
                
            if step % 1000 == 1:
                gt   = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(2):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            
            print('epoch:{} {}/{} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, info['loss_l1']))
            step += 1

        if epoch % 10 == 0:
            evaluate(model, val_data, step)

        if epoch % 20 == 0:
            model.save_model(f'{log_path}/flownet_{epoch:04d}.pkl')
            
    model.save_model(f'{log_path}/flownet_last.pkl')

def evaluate(model, val_data, step):
    loss_l1_list = []
    loss_cons_list = []
    loss_tea_list = []
    psnr_list = []
    lpips_list = []
    psnr_list_teacher = []
    
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        timestep = timestep.to(device, non_blocking=True)      
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False, timestep=timestep)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_cons_list.append(info['loss_cons'].cpu().numpy())
        for j in range(gt.shape[0]):
            lpips = loss_fn_alex(gt[j] * 2 - 1, pred[j] * 2 - 1).detach().cpu().data
            lpips_list.append(lpips)
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
            lpips_list.append(lpips)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0:
            for j in range(4):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), step, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), step, dataformats='HWC')
    
    writer_val.add_scalar('benchmark/psnr', np.array(psnr_list).mean(), step)
    writer_val.add_scalar('benchmark/psnr_teacher', np.array(psnr_list_teacher).mean(), step)
    writer_val.add_scalar('benchmark/lpips', np.array(lpips_list).mean(), step)
        
if __name__ == "__main__":
    seed = 124
    torch.cuda.set_device(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    args = parser.parse_args()
    model = Model()
    train(model)
        
