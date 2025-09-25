import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from rife import Model

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def main(model, prefix, videogen):
    frame = cv2.imread(os.path.join(prefix, videogen[0]), cv2.IMREAD_COLOR )
    h, w, _ = frame.shape
    tmp = 32
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    cnt = 0
    for s in range(1, args.step):
        print(s / args.step)

    for i in tqdm(range(0, len(videogen) - 1), desc="rending"):
        I0 = cv2.imread(os.path.join(args.imgs, videogen[i]), cv2.IMREAD_COLOR)
        cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), I0)
        cnt = cnt + 1
        I0 = torch.from_numpy(np.transpose(I0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I0 = F.pad(I0, padding)

        I1 = cv2.imread(os.path.join(args.imgs, videogen[i + 1]), cv2.IMREAD_COLOR)
        I1 = torch.from_numpy(np.transpose(I1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = F.pad(I1, padding)

        for s in range(1, args.step):
            mid, flow, mask = model.inference(I0, I1, s / args.step)
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            mid = mid[:h, :w]
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), mid)

            #flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy()
            #flow = flow2rgb(flow[i])
            #flow = (flow * 255.).astype(np.int8)
            #cv2.imwrite('vid_out/{:0>7d}_flow.png'.format(cnt), flow)

            #mask = mask.permute(0, 2, 3, 1).detach().cpu().numpy()
            #mask = (mask[0] * 255.).astype(np.int8)
            #cv2.imwrite('vid_out/{:0>7d}_mask.png'.format(cnt), mask)

            cnt = cnt + 1

    I1 = (((I1[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
    I1 = I1[:h, :w]
    cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), I1)
    cnt = cnt + 1

if __name__ == '__main__':
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--imgs', dest='imgs', type=str, default='imgs')
    parser.add_argument('--step', dest='step', type=int, default=2)
    parser.add_argument('--sft',  dest='sft',  type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()

    
    if args.sft == 0:
        model_name = 'weights/flownet.426.pkl'
        model.load_model(model_name, False)
    else:
        model_name = 'weights/flownet_last.pkl'
        model.load_model(model_name, True)

    print(f'load {model_name}')
    model.eval()
    model.device()

    videogen = []
    for f in os.listdir(args.imgs):
        if 'png' in f:
            videogen.append(f)
    videogen.sort(key= lambda x:x[:-4])

    main(model, args.imgs, videogen)