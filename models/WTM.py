import os
import torch
import torch.nn.functional as F
import numpy as np
from .wae import WAE
import sys
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import collections
import math
from torch.cuda.amp import autocast, GradScaler
import time
import torch.optim.lr_scheduler as lr_scheduler
import random


class WTM:
    def __init__(self, bow_dim=10000, n_topic=20, device=None, dropout=0.0, alpha=0.2):
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        self.wae = WAE(encode_dims=[bow_dim, 50, 30, n_topic], decode_dims=[n_topic, 50, bow_dim], dropout=dropout, device=device)
        self.wae.initialize()
        self.device = device
        self.alpha = np.full([n_topic, ], alpha)

        if device != None:
            self.wae = self.wae.to(device)

    def train(self, train_dataloader, num_epochs=100, save_dir=None, margin=0.05):
        optimizer = torch.optim.AdamW(self.wae.parameters(), lr=1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.01 if epoch < 200 else 0.001)

        start_epoch = 0
        for epoch in range(start_epoch, num_epochs):
            train_loss = 0
            train_rec_loss = 0
            train_cos_loss = 0
            train_kl_loss = 0
            train_mmd_loss = 0
            train_contras_loss = 0
            train_num = 0

            self.wae.train()


            for iter, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                bows = data[0].to(self.device)
                positive_sample = data[1].to(self.device)
                negative_sample = data[2].to(self.device)
                pos_num = data[3]
                neg_num = data[4]


                with autocast(enabled=False):
                    bows_recon, theta_q, weight = self.wae(bows)
                    _, theta_q_pos, _ = self.wae(positive_sample)
                    _, theta_q_neg, _ = self.wae(negative_sample)

                    theta_prior = self.wae.sample(alpha=self.alpha, batch_size=len(bows)).to(self.device)

                    bows_recon = F.softmax(bows_recon, dim=1)
                    cos_loss = (1 - cosine_similarity(bows_recon, bows, axis=1)).sum() / bows.shape[0]
                    kl_loss_calc = nn.KLDivLoss(reduction='batchmean', log_target=True)
                    kl_loss = kl_loss_calc(torch.log(bows_recon), torch.log(bows + 1e-6))
                    rec_loss = (cos_loss + kl_loss) * 0.5
                    mmd = self.wae.mmd_loss(theta_q.squeeze(), theta_prior, device=self.device, t=0.1) * 10.0
                    contras_loss = self.wae.contrastive_loss_triplet(theta_q, theta_q_pos, theta_q_neg, pos_num.to(self.device),
                                                                     neg_num.to(self.device), margin=margin, device=self.device)

                    loss = rec_loss + mmd + contras_loss


                loss.backward()
                optimizer.step()

                train_loss += loss.item() * bows.shape[0]
                train_rec_loss += rec_loss.item() * bows.shape[0]
                train_cos_loss += cos_loss.item() * bows.shape[0]
                train_kl_loss += kl_loss.item() * bows.shape[0]
                train_mmd_loss += mmd.item() * bows.shape[0]
                train_contras_loss += contras_loss.item() * bows.shape[0]
                train_num += bows.size(0)

            if epoch == num_epochs - 1:
                last_model = self.wae.state_dict()
            scheduler.step()

            print('Epoch {}/{} completed'.format(epoch + 1, num_epochs))
        if save_dir != None:
            torch.save(last_model, os.path.join(save_dir, 'last.pkl'))


    def inference_by_bow(self, spot_fea):
        with torch.no_grad():
            self.wae.eval()
            with autocast(enabled=False):
                theta = F.softmax(self.wae.encode(spot_fea.to(self.device)), dim=-1)
                bows_recon, _, _ = self.wae(spot_fea.to(self.device))
            return theta.detach().cpu().numpy(), bows_recon.detach().cpu().numpy()

    def theta_filter(self, result, thres=0.01):
        result[result < thres] = 0
        row_sum = result.sum(axis=1)
        result = result / row_sum[:, np.newaxis]

        return result

    def get_embed(self, dataloader):
        self.wae.eval()
        embed_lst = []
        recon_lst = []
        for data_batch in dataloader:
            embed, recon = self.inference_by_bow(data_batch)
            embed_lst.append(embed)
            if len(recon.shape) < 2:
                recon = recon[np.newaxis, :]
            recon_lst.append(recon)
        embed_lst = np.concatenate(embed_lst, axis=0)
        recon_lst = np.concatenate(recon_lst, axis=0)
        # embed_lst = self.theta_filter(embed_lst, thres=0.05)
        return embed_lst, recon_lst


    def get_topic_top_words(self, gene_names_list, top_k=5):
        self.wae.eval()
        decoder_w = self.wae.state_dict()['decoder.0.weight'].T.detach().cpu()
        beta = F.softmax(decoder_w, dim=1)
        params_new = -np.array(beta)
        top_gene_ids = np.argsort(params_new)[:, :top_k]
        top_gene_names = np.empty_like(top_gene_ids, dtype='U100')

        for i in range(top_gene_ids.shape[0]):
            for j in range(top_gene_ids.shape[1]):
                top_gene_names[i][j] = gene_names_list[top_gene_ids[i][j]]

        return top_gene_ids, top_gene_names, beta


    def calc_rare_cell_type_num(self, embeds, thres=0.05):
        proportion_mean = np.mean(embeds, axis=0)
        rare_num = np.sum(proportion_mean < thres)
        rare_cell_type = np.where(proportion_mean < thres)

        return rare_num, rare_cell_type


    def calc_perplexity(self, test_data, recon_data):  # test_data pp
        logsoftmax = torch.log_softmax(torch.from_numpy(recon_data), dim=1)
        J = -1.0 * torch.sum(torch.from_numpy(np.array(test_data)) * logsoftmax) / len(test_data)

        perplexity = math.exp(J)
        return perplexity


