import os
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader
from ..models import WTM
from .dataset import make_data_contrastive_square, make_data_contrastive_hexagon, make_data_contrastive_others, make_test_data_contrastive, train_Dataset_contrastive,  test_Dataset_contrastive
import random
import pandas as pd



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize_expr(data):
    data = data / (data.sum(axis=1, keepdims=True) + 1e-10)

    return data


def collate_fn(batch):
    spots, pos_samples, neg_samples, pos_num, neg_num = zip(*batch)

    return torch.cat(spots, dim=0), torch.cat(pos_samples, dim=0), torch.cat(neg_samples, dim=0), torch.tensor(pos_num), torch.tensor(neg_num)


def deconvolution(df_data, cell_type_num, spatial_mode, save_dir_name, pos_rate=0.05, neg_rate=0.05, alpha=0.2, margin=0.05, batch_size=64, num_epoch=500, num_workers=0, device_id=None, random_seed=0):
    setup_seed(random_seed)
    if device_id != None:
        device = torch.device('cuda:{}'.format(device_id))
    else:
        device = torch.device('cpu')
        print('It is recommended that you use GPU for acceleration, otherwise the running time will be very long!')

    print('\nData organization...')
    # df_expr = pd.read_csv(data_dir + 'spot_pos_and_expr.csv')
    df_expr = df_data
    spatial_fea_num = df_expr.shape[1] - 3
    df_expr.iloc[:, 3:] = normalize_expr(df_expr.iloc[:, 3:].to_numpy())

    if spatial_mode == 'square':
        train_data = make_data_contrastive_square(df_expr, pos_rate=pos_rate, neg_rate=neg_rate)
    elif spatial_mode == 'hexagon':
        train_data = make_data_contrastive_hexagon(df_expr, pos_rate=pos_rate, neg_rate=neg_rate)
    else:
        train_data = make_data_contrastive_others(df_expr, pos_rate=pos_rate, neg_rate=neg_rate)
    print('Total_spot_num:{}'.format(len(train_data)))
    test_data, pos_all = make_test_data_contrastive(df_expr)

    train_set = train_Dataset_contrastive(data=train_data)
    test_set = test_Dataset_contrastive(data=test_data)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print('Data organization completed.')

    begin_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('begin time:{}'.format(begin_time))

    print('\nDeconvolution begin...')
    model = WTM(bow_dim=spatial_fea_num, n_topic=cell_type_num, device=device, dropout=0.5, alpha=alpha)
    train_save_dir = "results_save/{}_{}/model_save/".format(save_dir_name, begin_time)
    os.makedirs(train_save_dir, exist_ok=True)
    model.train(train_dataloader=train_dataloader, num_epochs=num_epoch, save_dir=train_save_dir, margin=margin)
    embeds, recon = model.get_embed(dataloader=test_dataloader)
    spatial_gene_list = df_expr.columns[3:].tolist()
    top_gene_ids, top_gene_names, beta = model.get_topic_top_words(gene_names_list=spatial_gene_list, top_k=5)
    print('Deconvolution completed')

    df_deconvolution_results = pd.DataFrame(embeds, columns=[i for i in range(embeds.shape[1])])
    df_deconvolution_results = pd.concat([df_expr.iloc[:, :3], df_deconvolution_results], axis=1)
    df_beta = pd.DataFrame(beta, columns=spatial_gene_list, index=[i for i in range(embeds.shape[1])])
    df_beta = df_beta.reset_index()

    save_dir = "results_save/{}_{}/prediction_save/".format(save_dir_name, begin_time)
    os.makedirs(save_dir, exist_ok=True)
    df_deconvolution_results.to_csv(save_dir + 'pred.csv', index=False)
    df_beta.to_csv(save_dir + 'beta.csv', index=False)

    return df_deconvolution_results, df_beta, test_dataloader


def deconvolution_multi_ctn(df_data, cell_type_num, spatial_mode, save_dir_name, pos_rate=0.05, neg_rate=0.05, alpha=0.2, margin=0.05, batch_size=64, num_epoch=500, device_id=None, random_seed=0, num_workers=0):
    setup_seed(random_seed)
    if device_id != None:
        device = torch.device('cuda:{}'.format(device_id))
    else:
        device = torch.device('cpu')
        print('It is recommended that you use GPU for acceleration, otherwise the running time will be very long!')

    print('\nData organization...')

    df_expr = df_data
    spatial_fea_num = df_expr.shape[1] - 3
    df_expr.iloc[:, 3:] = normalize_expr(df_expr.iloc[:, 3:].to_numpy())

    if spatial_mode == 'square':
        train_data = make_data_contrastive_square(df_expr, pos_rate=pos_rate, neg_rate=neg_rate)
    elif spatial_mode == 'hexagon':
        train_data = make_data_contrastive_hexagon(df_expr, pos_rate=pos_rate, neg_rate=neg_rate)
    else:
        train_data = make_data_contrastive_others(df_expr, pos_rate=pos_rate, neg_rate=neg_rate)
    print('Total_spot_num:{}'.format(len(train_data)))
    test_data, pos_all = make_test_data_contrastive(df_expr)

    train_set = train_Dataset_contrastive(data=train_data)
    test_set = test_Dataset_contrastive(data=test_data)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print('Data organization completed.')

    begin_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('begin time:{}'.format(begin_time))

    parameter_adjustment = []
    begin_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    for cell_type_num_i in cell_type_num:
        setup_seed(random_seed)
        print(f'\nCell type number = {cell_type_num_i}, begin deconvolution....')
        os.makedirs("results_save/{}_{}/model_save/cell_type_num_{}".format(save_dir_name, begin_time, cell_type_num_i), exist_ok=True)
        model = WTM(bow_dim=spatial_fea_num, n_topic=cell_type_num_i, device=device, dropout=0.5, alpha=alpha)
        model.train(train_dataloader=train_dataloader, num_epochs=num_epoch, save_dir="results_save/{}_{}/model_save/cell_type_num_{}".format(save_dir_name, begin_time, cell_type_num_i), margin=margin)

        embeds, recons = model.get_embed(dataloader=test_dataloader)
        top_gene_ids, top_gene_names, beta = model.get_topic_top_words(gene_names_list=df_expr.columns.tolist()[3:], top_k=5)
        rare_num, rare_cell_type = model.calc_rare_cell_type_num(embeds, thres=0.05)
        ppl = model.calc_RD(test_data, recons)
        os.makedirs("results_save/{}_{}/prediction_save/cell_type_num_{}".format(save_dir_name, begin_time, cell_type_num_i), exist_ok=True)
        dict_i = {'cell_type_num': cell_type_num_i, 'rare_cell_type_num': rare_num, 'ppl': ppl}

        parameter_adjustment.append(dict_i)

        df = pd.DataFrame(parameter_adjustment)
        save_dir = 'results_save/{}_{}/'.format(save_dir_name, begin_time)
        df.to_csv(save_dir + '{}.csv'.format(save_dir_name), index=False)
        df_pred = pd.DataFrame(embeds.squeeze())
        df_pred.to_csv(save_dir + 'prediction_save/cell_type_num_{}/pred.csv'.format(cell_type_num_i), index=False)
        df_beta = pd.DataFrame(beta, columns=df_expr.columns[3:])
        df_beta.to_csv(save_dir + 'prediction_save/cell_type_num_{}/beta.csv'.format(cell_type_num_i), index=False)

    return df, test_dataloader
