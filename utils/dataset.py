import random
import numpy as np
import copy

import torch.utils.data as Data
import networkx as nx
from networkx import from_numpy_array
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import torch
from sklearn.metrics.pairwise import cosine_similarity
import math


def make_data_contrastive_square(df_expr, neg_rate=0.05, pos_rate=0.05):
    # obtain pos and expr info
    spot_pos_all = df_expr.iloc[:, 1:3].to_numpy()
    spot_info_all = df_expr.iloc[:, 3:].to_numpy()

    # calculate cosine matrix
    expr = copy.deepcopy(spot_info_all)
    similarity_matrix = cosine_similarity(expr)
    hist, bin_edges = np.histogram(similarity_matrix, bins=20, range=(0.0, 1.0))

    near_similarity_all = []
    indices_1_all = []
    for i, spot_pos_i in enumerate(spot_pos_all):
        indices_1 = np.where((np.abs(spot_pos_all[:, 0] - spot_pos_i[0]) <= 1) & (np.abs(spot_pos_all[:, 1] - spot_pos_i[1]) <= 1))[0]
        indices_1_all.append(indices_1)
        near_similarity_all += list(similarity_matrix[i, indices_1])
    sorted_lst = np.sort(near_similarity_all)[::-1]
    sorted_lst = sorted_lst[len(spot_pos_all):]
    top_thres = sorted_lst[int(len(sorted_lst) * pos_rate)]

    similarity_all = similarity_matrix.flatten()
    sigma = len(spot_pos_all) * neg_rate
    low_thres = np.sort(similarity_all)[int(len(spot_pos_all) * sigma)]

    # find positive and negative samples
    train_data = []
    pos_statistics = []
    neg_statistics = []

    if sigma > 200:
        neg_num = 200
    else:
        neg_num = int(sigma)
    for i in range(len(spot_pos_all)):
        # find positive samples
        indices_2 = np.where(similarity_matrix[i, :] > top_thres)[0]  # high expression similarity
        indices = np.intersect1d(indices_1_all[i], indices_2)
        pos_statistics.append(len(indices))
        positive_spot = np.array(spot_info_all)[indices, :]

        # find negative samples
        indices = np.where(similarity_matrix[i, :] <= low_thres)[0]
        if len(indices) >= 1 and len(indices) <= neg_num:
            neg_statistics.append(len(indices))
        elif len(indices) < 1:
            indices = np.argsort(similarity_matrix[i, :], axis=None)[0:1]
            neg_statistics.append(len(indices))
        else:
            indices = np.argsort(similarity_matrix[i, :], axis=None)[0:neg_num]
            neg_statistics.append(len(indices))
        negative_spot = np.array(spot_info_all)[indices, :]

        # save info
        train_data.append([spot_info_all[i], positive_spot, negative_spot])

    return train_data


def make_data_contrastive_hexagon(df_expr, neg_rate=0.05, pos_rate=0.05):
    # obtain pos and expr info
    spot_pos_all = df_expr.iloc[:, 1:3].to_numpy()
    spot_info_all = df_expr.iloc[:, 3:].to_numpy()

    # calculate cosine matrix
    expr = copy.deepcopy(np.array(spot_info_all))
    similarity_matrix = cosine_similarity(expr)
    hist, bin_edges = np.histogram(similarity_matrix, bins=20, range=(0.0, 1.0))

    near_similarity_all = []
    indices_1_all = []
    for i, spot_pos_i in enumerate(spot_pos_all):
        indices_1 = np.where((np.abs(spot_pos_all[:, 0] - spot_pos_i[0]) <= 1) & (np.abs(spot_pos_all[:, 1] - spot_pos_i[1]) <= 2))[0]
        indices_1_all.append(indices_1)
        near_similarity_all += list(similarity_matrix[i, indices_1])
    sorted_lst = np.sort(near_similarity_all)[::-1]
    sorted_lst = sorted_lst[len(spot_pos_all):]
    top_thres = sorted_lst[int(len(sorted_lst) * pos_rate)]

    similarity_all = similarity_matrix.flatten()
    sigma = len(spot_pos_all) * neg_rate
    low_thres = np.sort(similarity_all)[int(len(spot_pos_all) * sigma)]

    # find positive and negative samples
    train_data = []
    pos_statistics = []
    neg_statistics = []

    if sigma > 200:
        neg_num = 200
    else:
        neg_num = int(sigma)
    for i in range(len(spot_pos_all)):
        # find positive samples
        indices_2 = np.where(similarity_matrix[i, :] > top_thres)[0]  # high expression similarity
        indices = np.intersect1d(indices_1_all[i], indices_2)
        pos_statistics.append(len(indices))
        positive_spot = np.array(spot_info_all)[indices, :]

        # find negative samples
        indices = np.where(similarity_matrix[i, :] <= low_thres)[0]
        if len(indices) >= 1 and len(indices) <= neg_num:
            neg_statistics.append(len(indices))
        elif len(indices) < 1:
            indices = np.argsort(similarity_matrix[i, :], axis=None)[0:1]
            neg_statistics.append(len(indices))
        else:
            indices = np.argsort(similarity_matrix[i, :], axis=None)[0:neg_num]
            neg_statistics.append(len(indices))
        negative_spot = np.array(spot_info_all)[indices, :]

        # save info
        train_data.append([spot_info_all[i], positive_spot, negative_spot])

    return train_data


def make_data_contrastive_others(df_expr, neg_rate=0.05, pos_rate=0.05):
    # obtain pos and expr info
    spot_pos_all = df_expr.iloc[:, 1:3].to_numpy()
    spot_info_all = df_expr.iloc[:, 3:].to_numpy()

    # calculate cosine matrix
    expr = copy.deepcopy(spot_info_all)
    similarity_matrix = cosine_similarity(expr)
    hist, bin_edges = np.histogram(similarity_matrix, bins=20, range=(0.0, 1.0))

    distance = np.sum((spot_pos_all - spot_pos_all[:, np.newaxis]) ** 2, axis=2)
    indices_1_all = np.argpartition(distance, 11)[:, :11]
    near_similarity_all = similarity_matrix[np.arange(len(spot_pos_all))[:, np.newaxis], indices_1_all].flatten()
    sorted_lst = np.sort(near_similarity_all)[::-1]
    sorted_lst = sorted_lst[len(spot_pos_all):]
    top_thres = sorted_lst[int(len(sorted_lst) * pos_rate)]

    similarity_all = similarity_matrix.flatten()
    sigma = len(spot_pos_all) * neg_rate
    low_thres = np.sort(similarity_all)[int(len(spot_pos_all) * sigma)]

    # find positive and negative samples
    train_data = []
    pos_statistics = []
    neg_statistics = []

    if sigma > 200:
        neg_num = 200
    else:
        neg_num = int(sigma)

    for i in range(len(spot_pos_all)):
        # find positive samples
        indices_2 = np.where(similarity_matrix[i, :] > top_thres)[0]  # high expression similarity
        indices = np.intersect1d(indices_1_all[i], indices_2)
        pos_statistics.append(len(indices))
        positive_spot = spot_info_all[indices, :]

        # find negative samples
        indices = np.where(similarity_matrix[i, :] <= low_thres)[0]
        if len(indices) >= 1 and len(indices) <= neg_num:
            neg_statistics.append(len(indices))
        elif len(indices) < 1:
            indices = np.argsort(similarity_matrix[i, :], axis=None)[0:1]
            neg_statistics.append(len(indices))
        else:
            indices = np.argsort(similarity_matrix[i, :], axis=None)[0:neg_num]
            neg_statistics.append(len(indices))
        negative_spot = spot_info_all[indices, :]

        # save info
        train_data.append([spot_info_all[i], positive_spot, negative_spot])

    return train_data


def make_test_data_contrastive(df_expr):
    spot_pos_all = []
    spot_info_all = []

    # obtain pos and expr info
    for i in range(len(df_expr)):
        spot_pos_all.append(df_expr.iloc[i, 1:3].to_numpy().astype(float))
        spot_info_all.append(df_expr.iloc[i, 3:].to_numpy().astype(float))

    test_data = spot_info_all

    return test_data, spot_pos_all


class train_Dataset(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        spot = copy.deepcopy(self.data[index])
        spot = torch.from_numpy(spot).float()
        # patch_noise = add_noise(patch)

        return spot

    def __len__(self):
        return len(self.data)


class test_Dataset(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        spot = copy.deepcopy(self.data[index])
        spot = torch.from_numpy(spot).float()

        return spot

    def __len__(self):
        return len(self.data)


class train_Dataset_contrastive(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        spot = copy.deepcopy(self.data[index][0])
        positive_sample = copy.deepcopy(self.data[index][1])
        negative_sample = copy.deepcopy(self.data[index][2])
        spot = torch.from_numpy(spot).float().unsqueeze(0)
        positive_sample = torch.from_numpy(positive_sample).float()
        negative_sample = torch.from_numpy(negative_sample).float()
        # patch_noise = add_noise(patch)

        return spot, positive_sample, negative_sample, len(positive_sample), len(negative_sample)

    def __len__(self):
        return len(self.data)


class test_Dataset_contrastive(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        spot = copy.deepcopy(self.data[index]).astype(np.float32)
        spot = torch.from_numpy(spot)

        return spot

    def __len__(self):
        return len(self.data)



