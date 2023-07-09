'''
main for HyperCI
'''

import time
import argparse
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Model import HyperSCI, GraphSCI
import utils
import scipy.io as sio
from sklearn.linear_model import LinearRegression, Ridge
import data_preprocessing as dpp
import data_simulation as dsim

from scipy import sparse as sp
import scipy.io as sio
import csv
import torch_geometric.nn as gnn
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
import matplotlib

font_sz = 28
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams.update({'font.size': font_sz})

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='contact')  # contact, GoodReads Microsoft

parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1601, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--h_dim', type=int, default=25, help='dim of hidden units.')
parser.add_argument('--g_dim', type=int, default=32, help='dim of treatment representation.')
parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
parser.add_argument('--activate', type=int, default=0)
parser.add_argument('--normy', type=int, default=0)
parser.add_argument('--num_gnn_layer', type=int, default=1)
parser.add_argument('--n_out', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--phi_layer', type=int, default=1)
parser.add_argument('--skip', type=str, default='23', choices=['123', '3', '23', '13', '1'])
parser.add_argument('--graph_model', type=str, default='hypergraph', choices=['hypergraph', 'graph'])  # hypergraph: our model; graph: gcn based baseline
parser.add_argument('--graph_type', type=str, default='hypergraph', choices=['hypergraph', 'projected'])   # use hypergraph or projected graph
parser.add_argument('--index_type', type=str, default='hyper_index', choices=['hyper_index', 'graph_index'])  # graph_index for baseline
parser.add_argument('--path', type=str, default='../data/contact.mat')
parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--exp_name', type=str, default='ITE', choices=['ITE', 'LR', 'case', 'hypersize'])
parser.add_argument('--LR_name', type=str, default='S', choices=['S', 'T', 'T_agg'])  # linear regression: S-Learner, T-learner
parser.add_argument('--max_hyperedge_size', type=int, default=50,
                    help='only keep hyperedges with size no more than this value (only valid in hypersize experiment)')
parser.add_argument('--wass', type=float, default=1e-2)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def compute_loss(Y_true, treatments, results, idx_trn, idx_select):
    # binary
    y1_true = Y_true[1]
    y0_true = Y_true[0]
    rep = results['rep']
    y1_pred = results['y1_pred']
    y0_pred = results['y0_pred']
    yf_pred =  torch.where(treatments > 0, y1_pred, y0_pred)

    # balancing
    num_balance_max = 2000  # max num of instances used for balancing
    idx_balance = idx_select if len(idx_select) < num_balance_max else idx_select[: num_balance_max]
    rep_t1, rep_t0 = rep[idx_balance][(treatments[idx_balance] > 0).nonzero()], rep[idx_balance][(treatments[idx_balance] < 1).nonzero()]

    # wass1 distance
    dist, _ = utils.wasserstein(rep_t1, rep_t0, device, cuda=True)

    # potential outcome prediction
    YF = torch.where(treatments > 0, y1_true, y0_true)

    # norm y
    if args.normy:
        ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
        YF_select = (YF[idx_select] - ym) / ys
    else:
        YF_select = YF[idx_select]

    # loss: (Y-Y_hat)^2 + alpha * w-dist
    loss_mse = torch.nn.MSELoss()
    loss_y = loss_mse(yf_pred[idx_select], YF_select)

    loss = loss_y + args.wass * dist

    loss_result = {
        'loss': loss, 'loss_y': loss_y, 'loss_b': dist
    }

    return loss_result

def evaluate(Y_true, treatments, results, idx_trn, idx_select, keep_orin_ite=False):
    y1_true, y0_true = Y_true[1], Y_true[0]

    y1_pred = results['y1_pred']
    y0_pred = results['y0_pred']

    # potential outcome prediction
    YF = torch.where(treatments > 0, y1_true, y0_true)

    # norm y
    if args.normy:
        ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    ITE_pred = y1_pred - y0_pred
    ITE_true = y1_true - y0_true

    # metrics
    n_select = len(idx_select)
    ate = (torch.abs((ITE_pred[idx_select] - ITE_true[idx_select]).mean())).item()
    pehe = math.sqrt(((ITE_pred[idx_select] - ITE_true[idx_select]) * (ITE_pred[idx_select] - ITE_true[idx_select])).sum().data / n_select)

    RMSE_Y1 = torch.sqrt(torch.mean(torch.pow(y1_true[idx_select] - y1_pred[idx_select], 2))).item()
    RMSE_Y0 = torch.sqrt(torch.mean(torch.pow(y0_true[idx_select] - y0_pred[idx_select], 2))).item()

    eval_results = {'pehe': pehe, 'ate': ate, 'RMSE_Y1': RMSE_Y1, 'RMSE_Y0': RMSE_Y0}
    if keep_orin_ite:
        eval_results['ITE_pred'] = ITE_pred

    return eval_results

def report_info(epoch, time_begin, loss_results_train, eval_results_val, eval_results_tst):
    loss_train = loss_results_train['loss']
    loss_y = loss_results_train['loss_y']
    loss_b = loss_results_train['loss_b']
    pehe_val, ate_val = eval_results_val['pehe'], eval_results_val['ate']
    pehe_tst, ate_tst, RMSE_Y1_tst, RMSE_Y0_tst = eval_results_tst['pehe'], eval_results_tst['ate'], eval_results_tst['RMSE_Y1'], eval_results_tst['RMSE_Y0']

    print('Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'pehe_tst: {:.4f}'.format(pehe_tst),
            'ate_tst: {:.4f} '.format(ate_tst),
            'time: {:.4f}s'.format(time.time() - time_begin)
          )

def train(epochs, model, optimizer, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst):
    time_begin = time.time()
    print("start training!")

    for k in range(epochs):  # epoch
        model.train()
        optimizer.zero_grad()

        # forward
        results = model(features, treatments, hyperedge_index)

        # loss
        loss_results_train = compute_loss(Y_true, treatments, results, idx_trn, idx_trn)
        loss_train = loss_results_train['loss']

        loss_train.backward()
        optimizer.step()

        nn.utils.clip_grad_norm(model.parameters(), args.clip)

        if k % 100 == 0:
            # evaluate
            model.eval()
            results = model(features, treatments, hyperedge_index)
            eval_results_val = evaluate(Y_true, treatments, results, idx_trn, idx_val)
            eval_results_tst = evaluate(Y_true, treatments, results, idx_trn, idx_tst)

            report_info(k, time_begin, loss_results_train, eval_results_val, eval_results_tst)
    return

def test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_select, keep_orin_ite=False):
    model.eval()

    results = model(features, treatments, hyperedge_index)
    eval_results = evaluate(Y_true, treatments, results, idx_trn, idx_select, keep_orin_ite)

    pehe = eval_results['pehe']
    ate = eval_results['ate']
    RMSE_Y1_tst, RMSE_Y0_tst = eval_results['RMSE_Y1'], eval_results['RMSE_Y0']

    print('test results: ',
          'RMSE_Y1_tst: {:.4f}'.format(RMSE_Y1_tst),
          'RMSE_Y0_tst: {:.4f} '.format(RMSE_Y0_tst),
        'pehe_tst: {:.4f}'.format(pehe),
        'ate_tst: {:.4f} '.format(ate))

    return eval_results

def load_data(dataset, path, num_exp=10, graph_type='hypergraph', index_type='hyper_index', hyper_form_type='processed'):
    trn_rate = 0.6
    tst_rate = 0.2

    data = sio.loadmat(path)
    features, treatments, outcomes, Y_true, hyperedge_index = data['features'], data['treatments'][0], data['outcomes'][0], data['Y_true'], data['hyperedge_index']

    standarlize = True
    if standarlize:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(features)
        features = scaler.transform(features)

    print('loaded data from ', path)
    # print(dpp.hypergraph_stats(hyperedge_index, features.shape[0]))

    show_hyperedge_size = False
    if show_hyperedge_size:
        unique, frequency = np.unique(hyperedge_index[1], return_counts=True)
        print('hyperedge size: ', np.sort(frequency)[::-1][:100])  # top 100 hyperedge size
        dpp.draw_freq(frequency)

    if hyper_form_type == 'processed' and graph_type == 'projected' and args.exp_name != 'hypersize':
        hyperedge_index = utils.project_hypergraph(features.shape[0], hyperedge_index, type=index_type)

    idx_trn_list, idx_val_list, idx_tst_list = [], [], []
    idx_treated = np.where(treatments == 1)[0]
    idx_control = np.where(treatments == 0)[0]
    for i in range(num_exp):
        idx_treated_cur = idx_treated.copy()
        idx_control_cur = idx_control.copy()
        np.random.shuffle(idx_treated_cur)
        np.random.shuffle(idx_control_cur)

        idx_treated_trn = idx_treated_cur[: int(len(idx_treated) * trn_rate)]
        idx_control_trn = idx_control_cur[: int(len(idx_control) * trn_rate)]
        idx_trn_cur = np.concatenate([idx_treated_trn, idx_control_trn])
        idx_trn_cur = np.sort(idx_trn_cur)
        idx_trn_list.append(idx_trn_cur)

        idx_treated_tst = idx_treated_cur[int(len(idx_treated) * trn_rate): int(len(idx_treated) * trn_rate) + int(len(idx_treated) * tst_rate)]
        idx_control_tst = idx_control_cur[int(len(idx_control) * trn_rate): int(len(idx_control) * trn_rate) + int(len(idx_control) * tst_rate)]
        idx_tst_cur = np.concatenate([idx_treated_tst, idx_control_tst])
        idx_tst_cur = np.sort(idx_tst_cur)
        idx_tst_list.append(idx_tst_cur)
        idx_treated_val = idx_treated_cur[int(len(idx_treated) * trn_rate) + int(len(idx_treated) * tst_rate):]
        idx_control_val = idx_control_cur[int(len(idx_control) * trn_rate) + int(len(idx_control) * tst_rate):]
        idx_val_cur = np.concatenate([idx_treated_val, idx_control_val])
        idx_val_cur = np.sort(idx_val_cur)
        idx_val_list.append(idx_val_cur)

    # tensor
    features = torch.FloatTensor(features)
    treatments = torch.FloatTensor(treatments)
    Y_true = torch.FloatTensor(Y_true)
    outcomes = torch.FloatTensor(outcomes)

    if hyper_form_type == 'processed' and graph_type == 'projected' and index_type == 'graph_index':
        hyperedge_index = hyperedge_index.nonzero()  # sparse adjacency matrix -> edge index
    if hyper_form_type == 'processed':
        hyperedge_index = torch.LongTensor(hyperedge_index)
    idx_trn_list = [torch.LongTensor(id) for id in idx_trn_list]
    idx_val_list = [torch.LongTensor(id) for id in idx_val_list]
    idx_tst_list = [torch.LongTensor(id) for id in idx_tst_list]

    return features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list


def baseline_LR(features, treatment, outcome, Y_true, idx_trn, idx_val, idx_tst, hyperedge_index=None):
    # t-leaner
    if args.LR_name == 'T_agg':
        import data_simulation as sim
        features_agg = sim.agg_features(features, hyperedge_index, treatment, alpha=1.0)
        features = np.concatenate([features, features_agg], axis=1)

    if args.LR_name == 'T' or args.LR_name == 'T_agg':
        model_1 = LinearRegression()
        model_0 = LinearRegression()
        idx_treated_trn = np.where(treatment[idx_trn] == 1)
        idx_control_trn = np.where(treatment[idx_trn] == 0)

        model_1.fit(features[idx_trn[idx_treated_trn]], outcome[idx_trn[idx_treated_trn]])
        model_0.fit(features[idx_trn[idx_control_trn]], outcome[idx_trn[idx_control_trn]])

        y_pred1_tst = model_1.predict(features[idx_tst])
        y_pred0_tst = model_0.predict(features[idx_tst])

    # s-learner
    elif args.LR_name == 'S':
        model_t = LinearRegression()
        features_t = np.concatenate([features, treatment.reshape(-1, 1)], axis=1)
        model_t.fit(features_t[idx_trn], outcome[idx_trn])

        y_pred1_tst = model_t.predict(np.concatenate([features[idx_tst], np.ones((len(idx_tst), 1))], axis=1))
        y_pred0_tst = model_t.predict(np.concatenate([features[idx_tst], np.zeros((len(idx_tst), 1))], axis=1))

    y1_true_tst = Y_true[1][idx_tst]
    y0_true_tst = Y_true[0][idx_tst]

    # test
    ITE_pred_tst = y_pred1_tst - y_pred0_tst
    ITE_true_tst = y1_true_tst - y0_true_tst

    n_select = len(idx_tst)
    ate = np.abs((ITE_pred_tst - ITE_true_tst).mean())
    pehe = math.sqrt(((ITE_pred_tst - ITE_true_tst) * (
                ITE_pred_tst - ITE_true_tst)).sum() / n_select)
    RMSE_Y1 = math.sqrt(np.mean(np.power(y_pred1_tst - y1_true_tst, 2)))
    RMSE_Y0 = math.sqrt(np.mean(np.power(y_pred0_tst - y0_true_tst, 2)))

    eval_results = {'pehe': pehe, 'ate': ate, 'RMSE_Y1': RMSE_Y1, 'RMSE_Y0': RMSE_Y0}

    return eval_results

def data_statistics(features, treatments, outcomes, Y_true):
    y_obs = torch.where(treatments > 0, Y_true[1], Y_true[0])
    print('ITE ', torch.mean(Y_true[1]-Y_true[0]), torch.std(Y_true[1]-Y_true[0]))
    print('y_obs ', torch.mean(y_obs), torch.std(y_obs))
    print('outcomes ',torch.mean(outcomes), torch.std(outcomes))
    return

def experiment_LR(features, treatment, outcome, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, exp_num=3):
    t_begin = time.time()
    results_all = {'pehe': [], 'ate': [], 'RMSE_Y1': [], 'RMSE_Y0': []}

    for i_exp in range(0, exp_num):  # 10 runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        eval_results_tst = baseline_LR(features.numpy(), treatment.numpy(), outcome.numpy(), Y_true.numpy(), idx_trn.numpy(), idx_val.numpy(), idx_tst.numpy(), hyperedge_index=hyperedge_index.numpy())

        results_all['pehe'].append(eval_results_tst['pehe'])
        results_all['ate'].append(eval_results_tst['ate'])

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=np.float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=np.float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=np.float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=np.float))


    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    return


def experiment_ite(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, exp_num=3):
    t_begin = time.time()

    results_all = {'pehe': [], 'ate': [], 'RMSE_Y1': [], 'RMSE_Y0': []}

    for i_exp in range(0, exp_num):  # runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        # set model
        if args.graph_model == 'hypergraph':
            model = HyperSCI(args, x_dim=features.shape[1])
        elif args.graph_model == 'graph':
            model = GraphSCI(args, x_dim=features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # cuda
        if args.cuda:
            model = model.to(device)
            features = features.to(device)
            treatments = treatments.to(device)
            outcomes = outcomes.to(device)
            Y_true = Y_true.to(device)
            hyperedge_index = hyperedge_index.to(device)
            # if hyperedge_attr is not None:
            #     hyperedge_attr = hyperedge_attr.to(device)
            idx_trn_list = [id.to(device) for id in idx_trn_list]
            idx_val_list = [id.to(device) for id in idx_val_list]
            idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, model, optimizer, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst)
        eval_results_tst = test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_tst)

        results_all['pehe'].append(eval_results_tst['pehe'])
        results_all['ate'].append(eval_results_tst['ate'])
        results_all['RMSE_Y1'].append(eval_results_tst['RMSE_Y1'])
        results_all['RMSE_Y0'].append(eval_results_tst['RMSE_Y0'])
        # break  # if you just need one run

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=np.float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=np.float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=np.float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=np.float))

    results_all['average_rmse_y1'] = np.mean(np.array(results_all['RMSE_Y1'], dtype=np.float))
    results_all['std_rmse_y1'] = np.std(np.array(results_all['RMSE_Y1'], dtype=np.float))
    results_all['average_rmse_y0'] = np.mean(np.array(results_all['RMSE_Y0'], dtype=np.float))
    results_all['std_rmse_y0'] = np.std(np.array(results_all['RMSE_Y0'], dtype=np.float))

    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    return

# only keep hyperedges with size < max_hyperedge_size
def modify_hypergraph(hyperedge_index, max_hyperedge_size):
    idx_delete = []
    j = 0
    while j < len(hyperedge_index[1]):
        u = j
        while u < len(hyperedge_index[1]) and hyperedge_index[1][u] == hyperedge_index[1][j]:
            u += 1
        edge_size = u - j
        if edge_size > max_hyperedge_size:
            idx_delete += [i for i in range(j, j+edge_size)]
        j += edge_size

    # delete
    idx_select = list(set(range(len(hyperedge_index[1]))) - set(idx_delete))
    hyperedge_index = hyperedge_index[:, idx_select]

    # update edge index
    j = 0
    last = -1
    while j < len(hyperedge_index[1]):
        while j < len(hyperedge_index[1]) and hyperedge_index[1][j] == last + 1:
            j += 1
        if j != len(hyperedge_index[1]):  # not the end
            start = j
            new = hyperedge_index[1][j]
            while j < len(hyperedge_index[1]) and hyperedge_index[1][j] == new:
                j += 1
            hyperedge_index[1][start: j] = last + 1
        last += 1
    return hyperedge_index

def experiment_hypersize(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, max_hyperedge_size=2, exp_num=3):
    print('running experiment on hypergraph which removes higheredge with size more than ', max_hyperedge_size)
    t_begin = time.time()
    results_all = {'pehe': [], 'ate': []}

    # only keep hyperedges with size no more than k
    hyperedge_index = modify_hypergraph(hyperedge_index, max_hyperedge_size)

    if args.graph_model == 'graph':
        hyperedge_index = utils.project_hypergraph(features.shape[0], hyperedge_index, type=args.index_type)  # hypergraph->graph
        hyperedge_index = hyperedge_index.nonzero()  # sparse adjacency matrix -> edge index
        hyperedge_index = torch.LongTensor(hyperedge_index)

    for i_exp in range(0, exp_num):  # 10 runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        # set model
        if args.graph_model == 'hypergraph':
            model = HyperSCI(args, x_dim=features.shape[1])
        elif args.graph_model == 'graph':
            model = GraphSCI(args, x_dim=features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # cuda
        if args.cuda:
            model = model.to(device)
            features = features.to(device)
            treatments = treatments.to(device)
            outcomes = outcomes.to(device)
            Y_true = Y_true.to(device)
            hyperedge_index = hyperedge_index.to(device)
            # if hyperedge_attr is not None:
            #     hyperedge_attr = hyperedge_attr.to(device)
            idx_trn_list = [id.to(device) for id in idx_trn_list]
            idx_val_list = [id.to(device) for id in idx_val_list]
            idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, model, optimizer, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst)
        eval_results_tst = test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_tst)

        results_all['pehe'].append(eval_results_tst['pehe'])
        results_all['ate'].append(eval_results_tst['ate'])

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=np.float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=np.float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=np.float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=np.float))

    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    return

def compare_ite_diff(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, exp_num=3, type='none', postfix = ''):
    t_begin = time.time()

    results_all = {'ITE_diff': []}

    assert type == 'none' or type == 'projected'
    if type == 'none':
        hyperedge_index_weak = torch.LongTensor([range(len(features)), range(len(features))])  # no/weak structure information
    elif type == 'projected':
        hyperedge_index_weak = utils.project_hypergraph(features.shape[0], hyperedge_index, type='hyper_index')  # projected
        hyperedge_index_weak = torch.LongTensor(hyperedge_index_weak)

    idx_all = torch.LongTensor(range(len(features)))
    if args.cuda:
        idx_all = idx_all.to(device)
        hyperedge_index_weak = hyperedge_index_weak.to(device)

    for i_exp in range(0, exp_num):  # 10 runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        # set model
        if args.graph_model == 'hypergraph':
            model = HyperSCI(args, x_dim=features.shape[1])
        elif args.graph_model == 'graph':
            model = GraphSCI(args, x_dim=features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # cuda
        if args.cuda:
            model = model.to(device)
            features = features.to(device)
            treatments = treatments.to(device)
            outcomes = outcomes.to(device)
            Y_true = Y_true.to(device)
            hyperedge_index = hyperedge_index.to(device)
            # if hyperedge_attr is not None:
            #     hyperedge_attr = hyperedge_attr.to(device)
            idx_trn_list = [id.to(device) for id in idx_trn_list]
            idx_val_list = [id.to(device) for id in idx_val_list]
            idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, model, optimizer, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst)
        eval_results_all = test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_all,
                                keep_orin_ite=True)
        eval_results_all_weak = test(model, features, treatments, hyperedge_index_weak, Y_true, idx_trn, idx_all,
                                     keep_orin_ite=True)

        results_all['ITE_diff'].append(
            (eval_results_all['ITE_pred'] - eval_results_all_weak['ITE_pred']).view(1, -1))  # 1 x n

        # break  # !!!!!!!!!!!!!

    ite_diff = torch.cat(results_all['ITE_diff'], dim=0)  # exp_num x n
    results_all['average_ITE_diff'] = torch.mean(ite_diff, dim=0).cpu().detach().numpy().reshape(-1)  # n

    print('mean ite diff: ', np.mean(results_all['average_ITE_diff']), ' std: ',
          np.std(results_all['average_ITE_diff']))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    dpp.draw_freq(results_all['average_ITE_diff'])

    # save into files
    save_flag = True
    if save_flag:
        filename = '../data/goodreads_ite_diff_'+type+postfix+'.pickle'

        data_save = {'ite_diff': results_all['average_ITE_diff']}
        with open(filename, 'wb') as f:
            pickle.dump(data_save, f)
        print('saved file ', filename)

    return results_all['average_ITE_diff']

def query_hyper_statistics(features, treatments, outcomes, Y_true, hyperedge_index, types):
    hyperedge_index_np = hyperedge_index.cpu().detach().numpy()
    results = {}
    if 'treated_ratio' in types:
        results['treated_ratio'] = []
    if 'neighbor_num' in types:
        results['neighbor_num'] = []

    for i in range(features.shape[0]):
        neighbors_i = dsim.search_neighbor_hypergraph(i, hyperedge_index_np)  # not include itself
        if 'treated_ratio' in types:
            if len(neighbors_i) > 0:
                ti = treatments[i]
                t_neighbor = treatments[neighbors_i]
                equal_num = torch.where(t_neighbor == ti, 1.0, 0.0).sum()
                ratio = float(equal_num + 1) / (len(t_neighbor) + 1)  # +1, itself
                results['treated_ratio'].append(ratio)
            else:
                results['treated_ratio'].append(1)

        if 'neighbor_num' in types:
            results['neighbor_num'].append(len(neighbors_i))

    return results

def toDiscreteAxis(values, numOfBins=10, min_value=None, max_value=None):
    if min_value is None:
        min_value = min(values)
    if max_value is None:
        max_value = max(values)
    axis = []
    for i in range(numOfBins):
        axis.append(min_value + i * (max_value-min_value)/numOfBins)
    return axis, min_value, max_value

def experiment_case(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, exp_num=3, type_case='None', postfix=''):
    # heatmap
    with open('../data/goodreads_case_'+type_case+postfix+'.pickle', 'rb') as f:
        data_save = pickle.load(f)
    average_ITE_diff = np.sqrt(np.square(data_save['ite_diff']))
    treated_ratio = data_save['treated_ratio']  # list of
    neighbor_num = data_save['neighbor_num']

    #dpp.draw_freq(treated_ratio)
    #dpp.draw_freq(neighbor_num)

    bin_x = 6
    bin_y = 6

    ax_t, min_t, max_t = toDiscreteAxis(treated_ratio, bin_x)
    ax_n, min_n, max_n = toDiscreteAxis(neighbor_num, bin_y, max_value=30)

    n = len(average_ITE_diff)
    data_num_matrix = np.zeros((bin_x, bin_y))
    data_diff_matrix = np.zeros((bin_x, bin_y))
    idx_matrix = {str(i)+'_'+str(j): [] for i in range(bin_x) for j in range(bin_y)}

    for i in range(n):
        if treated_ratio[i] >= max_t:
            idx_x = bin_x - 1
        else:
            idx_x = int((treated_ratio[i] - min_t) / ((max_t - min_t) / bin_x))

        if neighbor_num[i] >= max_n:
            idx_y = bin_y - 1
        else:
            idx_y = int((neighbor_num[i] - min_n) / ((max_n - min_n) / bin_y))

        idx_matrix[str(idx_x)+'_'+str(idx_y)].append(i)
        data_num_matrix[idx_x][idx_y] += 1
        data_diff_matrix[idx_x][idx_y] += average_ITE_diff[i]

    norm_diff = data_diff_matrix / (data_num_matrix + 1)
    norm_diff_draw = norm_diff.copy()
    for i in range(len(norm_diff_draw)):
        norm_diff_draw[i] = norm_diff[len(norm_diff_draw) -1 - i]
    plt.imshow(norm_diff_draw, cmap='viridis')
    xlist = [round(((i+1) * (max_n - min_n) / bin_x)) for i in range(bin_x)]
    plt.xticks(np.arange(bin_x), xlist)
    ylist_orin = [round(((i+0.5)  * (max_t - min_t) / bin_y), 1) for i in range(bin_y)]
    ylist = ylist_orin.copy()
    ylist = [ylist_orin[len(ylist_orin) - 1 - i] for i in range(len(ylist_orin))]
    plt.yticks(np.arange(bin_y), ylist)
    plt.xlabel(r"$|\mathcal{N}_{(i)}|$")
    plt.ylabel(r"$r(i)$")

    cbar=plt.colorbar()
    cbar.ax.locator_params(nbins=5)
    plt.savefig("./" + 'case_' + type_case + '_' + postfix + '.pdf', bbox_inches='tight')
    plt.show()

    # book_select, meta info
    update_book_meta = False
    if update_book_meta:
        with open('../data/goodreads_select.pickle', 'rb') as f:
            data_select = pickle.load(f)
        books_select, authors_select = data_select['books_select'], data_select['authors_select']

        meta_result = dpp.load_goodreads_select_meta('../data/goodreads_books_children.json', books_select, authors_select)
        titles = meta_result['title']
        authors = meta_result['authors']
        for i in range(bin_x):
            for j in range(bin_y):
                idx_ij = idx_matrix[str(i)+'_'+str(j)]
                data_ij = []
                for book_idx in idx_ij:
                    data_ij.append({'id': book_idx, 'asin': books_select[book_idx], 'title': titles[book_idx], 'authors': authors[book_idx], 'treated_ratio': treated_ratio[book_idx], 'neighbor_num': neighbor_num[book_idx], 'ite_diff': str(average_ITE_diff[book_idx])})
                with open('../data/GoodReads_meta_'+str(i)+'_'+str(j)+'.json', 'w') as outfile:
                    json.dump(data_ij, outfile)

    return


if __name__ == '__main__':
    exp_num = 3
    if args.graph_model == 'graph':
        args.graph_type = 'projected'
        args.index_type = 'graph_index'

    print('exp_name: ', args.exp_name, ' graph_model: ', args.graph_model, ' encoder:', args.encoder, ' graph_type: ', args.graph_type, ' index_type: ', args.index_type)
    if args.exp_name == 'hypersize' and args.graph_model == 'graph':
        features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list = load_data(
            args.dataset, args.path, graph_type=args.graph_type, index_type=args.index_type, hyper_form_type='old')
    else:
        features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list = load_data(args.dataset, args.path, graph_type=args.graph_type, index_type=args.index_type)  # return tensors

    # =========  Experiment 1: compare with baselines ============
    if args.exp_name == 'LR':
        experiment_LR(features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, exp_num=exp_num)
    elif args.exp_name == 'ITE':
        experiment_ite(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, exp_num=exp_num)
    elif args.exp_name == 'hypersize':
        # ========== Experiment 2: only keep hyperedges with size no more than k ========
        max_hyperedge_size = args.max_hyperedge_size
        experiment_hypersize(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, max_hyperedge_size=max_hyperedge_size, exp_num=exp_num)
    elif args.exp_name == 'case':
        # ========= Experimet 3: case study ==============
        type_case = 'projected'  # 'none', 'projected'
        postfix = ''  # _realY, ''
        experiment_case(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list, exp_num=3, type_case=type_case, postfix=postfix)

