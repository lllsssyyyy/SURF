import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time
import multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence


class WAE(nn.Module):
    def __init__(self, encode_dims=[2000, 1024, 512, 20], decode_dims=[20, 512, 2000], dropout=0.0, device=None):
        super(WAE, self).__init__()

        self.device = device

        # encoder_linear
        self.enc_linear = nn.Sequential(
            nn.Linear(encode_dims[0], 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, encode_dims[-1]),
        )

        self.z_dim = encode_dims[-1]

        self.decoder = nn.Sequential(
            nn.Linear(encode_dims[-1], decode_dims[2]),
        )

        self.celltype_weight = nn.Parameter(torch.ones(encode_dims[-1]))

    def encode(self, spot_fea):
        hid = self.enc_linear(spot_fea)

        return hid

    def decode(self, theta):
        hid = self.decoder(theta)
        return hid

    def forward(self, spot_fea):
        z = self.encode(spot_fea)
        theta = F.softmax(z, dim=1)
        x_reconst = self.decode(theta.squeeze())
        return x_reconst, theta.squeeze(), F.softmax(self.celltype_weight, dim=0)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight.data)

    def sample(self, alpha, batch_size=256):
        z_true = np.random.dirichlet(alpha, size=batch_size)
        z_true = torch.from_numpy(z_true).float()
        return z_true

    def mmd_loss(self, x, y, device, t=0.1, kernel='diffusion'):
        '''
        computes the mmd loss with information diffusion kernel
        :param x: batch_size * latent dimension
        :param y:
        :param t:
        :return:
        '''
        eps = 1e-6
        n = x.shape[0]
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(i+1, n):
                    sum_xx = sum_xx + torch.norm(x[i]-x[j], p=1).to(device)
            sum_xx = sum_xx / (n * (n-1))

            sum_yy = torch.zeros(1).to(device)
            for i in range(y.shape[0]):
                for j in range(i+1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

            sum_xy = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())

            def diffusion_kernel(a, tmpt):
                return torch.exp(-torch.acos(a).pow(2) / tmpt)

            off_diag = 1 - torch.eye(n).to(device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t)
            sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
            sum_xy = 2 * k_xy.sum() / (n * n)
            # lamb = (4 * 3.1415926 * t) ** (-d/2)
        return (sum_xx + sum_yy - sum_xy)


    def contrastive_loss_cos(self, data, pos_data, neg_data, pos_num, neg_num, tau=1):
        cos = torch.nn.CosineSimilarity(dim=1)
        pos_data = torch.cat(pos_data, dim=0)
        neg_data = torch.cat(neg_data, dim=0)

        # normalize
        norm_value = torch.ones(data.shape[1]).to(self.device)
        # norm_value, _ = torch.max(torch.cat((data, pos_data, neg_data), dim=0), dim=0)
        # norm_value = torch.sum(torch.cat((data, pos_data, neg_data), dim=0), dim=0)
        data_ = data / norm_value.unsqueeze(0)
        pos_data_ = pos_data / norm_value.unsqueeze(0)
        neg_data_ = neg_data / norm_value.unsqueeze(0)

        data_for_pos_ = data_.repeat_interleave(pos_num, dim=0)
        data_for_neg_ = data_.repeat_interleave(neg_num, dim=0)

        pos_ = torch.exp(cos(data_for_pos_, pos_data_) / tau)
        neg_ = torch.exp(cos(data_for_neg_, neg_data_) / tau)
        pos_ = list(torch.split(pos_, list(pos_num), dim=0))
        neg_ = list(torch.split(neg_, list(neg_num), dim=0))

        loss = [(- torch.log(pos_i.mean() / (pos_i.mean() + neg_i.mean()))) for pos_i, neg_i in zip(pos_, neg_)]

        return sum(loss) / len(loss)

    def weighted_cos(self, A, B):
        """
        A: 64 * 1 * 9
        B: 64 * n * 9
        """
        w = F.softmax(self.celltype_weight, dim=0)
        dot_product = torch.sum(A * B * w, dim=2)
        norm_A = torch.norm(A, dim=2)
        norm_B = torch.norm(B, dim=2)
        weighted_cos = dot_product / (norm_A * norm_B + 1e-6)

        return weighted_cos

    def contrastive_loss_triplet(self, data, pos_data, neg_data, pos_num, neg_num, margin, device):
        start_time = time.time()
        pos_data_ = torch.split(pos_data, list(pos_num), dim=0)
        neg_data_ = torch.split(neg_data, list(neg_num), dim=0)

        pos_data__ = pad_sequence(pos_data_, batch_first=True)
        neg_data__ = pad_sequence(neg_data_, batch_first=True)

        data_for_pos_ = data.unsqueeze(1)
        data_for_neg_ = data.unsqueeze(1)
        pos_cos = self.weighted_cos(data_for_pos_, pos_data__)
        neg_cos = self.weighted_cos(data_for_neg_, neg_data__)

        loss = F.relu(- pos_cos.sum(dim=1) / pos_num + neg_cos.sum(dim=1) / neg_num + margin)

        end_time = time.time()

        return loss.mean()


