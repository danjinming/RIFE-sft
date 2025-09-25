import torch
import numpy as np
from torch.optim import AdamW
from loss import VGGPerceptualLoss, SSIM
from ifnet import *
device = torch.device("cuda")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
        
def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Model:
    def __init__(self):
        self.flownet_update = IFNet()
        self.flownet = IFNet()
        self.optimG = AdamW(self.flownet_update.parameters(), lr=1e-6, weight_decay=1e-2)
        self.ss = SSIM()
        self.vgg = VGGPerceptualLoss().to(device)
        self.encode_target = Head()
        self.device()

        hard_update(self.encode_target, self.flownet_update.encode)
        hard_update(self.flownet, self.flownet_update)

    def train(self):
        self.flownet_update.train()

    def eval(self):
        self.flownet_update.eval()

    def device(self):
        self.encode_target.to(device)
        self.flownet_update.to(device)
        self.flownet.to(device)

    def load_model(self, path, infer=False):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        
        param = torch.load(path)
        if not infer:
            param = convert(param)
        result = self.flownet_update.load_state_dict(param, False)
        hard_update(self.flownet, self.flownet_update)
        hard_update(self.encode_target, self.flownet.encode)
        return result
        
    def save_model(self, path):
        torch.save(self.flownet.state_dict(), path)

    def encode_loss(self, X, Y):
        loss = 0
        X = self.encode_target(X, True)
        Y = self.encode_target(Y, True)
        for i in range(4):
            loss += (X[i] - Y[i].detach()).abs().mean()
        return loss
    
    def inference(self, I0, I1, timestep):
        scale = [16, 8, 4, 2, 1]
        flow, mask, merged, _, _, _ = self.flownet_update(torch.cat((I0, I1), 1), timestep, scale=scale)
        return merged[-1], flow[-1], mask
        
    def update(self, imgs, gt, learning_rate=0, training=True, timestep=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [16, 8, 4, 2, 1]
        p = np.random.uniform(0, 1)
        if training:
            if p < 0.3:
                scale = [8, 4, 2, 2, 1]
            elif p < 0.6:
                scale = [4, 4, 2, 2, 1]
        flow, mask, merged, teacher_res, loss_cons, loss_time = self.flownet_update(torch.cat((imgs, gt), 1), timestep, scale=scale)
        loss_l1 = 0
        for i in range(5):
            loss_l1 *= 0.8
            loss_l1 += (merged[i] - gt).abs().mean()
        loss_l1 *= 0.1
        loss_tea = ((teacher_res[0][0] - gt).abs().mean() + (teacher_res[0][1] - gt).abs().mean()) * 0.1
        loss_cons += ((flow[-1] ** 2 + 1e-6).sum(1) ** 0.5).mean() * 1e-5
        loss_encode = 0
        for i in range(5):
            loss_encode *= 0.8
            loss_encode += self.encode_loss(merged[i], gt)
        loss_encode += self.encode_loss(teacher_res[0][0], gt) + self.encode_loss(teacher_res[0][1], gt)
        loss_encode *= 0.1
        loss_vgg, loss_gram = self.vgg(merged[-1], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = (loss_vgg + loss_encode + loss_gram) + loss_tea + loss_cons + loss_l1 - self.ss(merged[-1], gt) * 0.1 + loss_time
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.flownet_update.parameters(), 1.0)
            self.optimG.step()
            soft_update(self.encode_target, self.flownet_update.encode, 0.001)
            soft_update(self.flownet, self.flownet_update, 0.001)
            flow_teacher = teacher_res[1][0]
        else:
            flow_teacher = flow[-1]
        return merged[-1], {
            'merged_tea': teacher_res[0][0],
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[3][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_encode': loss_encode,
            'loss_vgg': loss_vgg,
            'loss_gram': loss_gram,
            'loss_cons': loss_cons,
            'loss_time': loss_time
            }


if __name__ == "__main__":
    path = 'weights/flownet_last.pkl'
    model = Model()
    result = model.load_model(path, True)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)