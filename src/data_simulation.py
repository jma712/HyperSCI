import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import collections
import os

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from sklearn.pipeline import Pipeline
import data_preprocessing as dpp
import json
import pickle
import math

import scipy.io as sio

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def search_neighbor_hypergraph(node, hyperedge_index):
    # return a list containing all neighbors (exist K times if in K same hyperedges) of node (NOT include itself)
    arg_idx = np.argwhere(hyperedge_index[0] == node).reshape(-1)
    edges = hyperedge_index[1][arg_idx]  # edges which include the node
    neighbors_idx = np.argwhere(np.isin(hyperedge_index[1], edges))   # all the neighbors on the edge
    neighbors_idx_noself =np.argwhere(np.isin(neighbors_idx, arg_idx).reshape(-1)).reshape(-1)
    neighbors_idx = np.delete(neighbors_idx, neighbors_idx_noself)  # remove node itself
    neighbors = hyperedge_index[0][neighbors_idx]
    return neighbors

# return a dictionary: {hyperedge_id: nodes}
def search_neighbor_hyperedge(node, hyperedge_index):
    arg_idx = np.argwhere(hyperedge_index[0] == node)
    edges = hyperedge_index[1][arg_idx]  # edges which include the node
    edges = np.unique(edges)

    hyperedge_neighbors = {}
    for ei in range(len(edges)):
        edge_id = edges[ei]
        neighbors_idx = np.argwhere(np.isin(hyperedge_index[1], [edge_id])).reshape(-1)  # all the neighbors on the edge, including itself
        neighbors = hyperedge_index[0][neighbors_idx]
        hyperedge_neighbors[edge_id] = neighbors
    return hyperedge_neighbors

def non_linear(x, type='raw'):
    if type == 'sigmoid':
        nl_x = 1 / (1 + np.exp(-x)) - 0.5
    if type == 'tanh':
        nl_x = np.tanh(x)
    elif type == 'raw':
        nl_x = x
    elif type == 'leaky_relu':
        nl_x = np.where(x > 0, x, x * 0.01)
    return nl_x


def simulate_outcome_2(features, hyperedge_index, treatments, type='linear', alpha=1.0, beta=1.0, nonlinear_type='raw'):
    n = features.shape[0]
    dim_size = features.shape[1]
    norm_type = 'size'  # size, no, fix, size_log, pow
    norm_value = 5
    norm_pow = 2

    agg_type = 'mean'
    agg_value = 50
    y_C = 1.0

    if type == 'linear':
        # 1. f_y00
        eps = np.random.normal(0.0, 0.001, size=1)
        W0 = np.random.normal(loc=0.0, scale=1.0, size=dim_size)
        #f_y00 = np.dot(features, W0) + eps  # f_{𝑦, 0}(x_𝑖) = W0 x_i;    (n x d) x (d x 1) -> n x 1
        f_y00 = 0.0  #
        # 2. ITE: f_t = t(\delta W x X)
        c_ft = 5
        W_t = np.random.normal(loc=0.05, scale=1.0, size=dim_size)  # contact: 0.05， GR: 1.0, 0.05
        ites = np.dot(features, W_t).reshape(-1) + c_ft  # (n x d) x (d x 1) -> n
        f_t = np.multiply(treatments, ites)  # n

        # 3. Spillover effect: s = aggregate(t_j * \tau(x_j)) = aggregate(t_j * (W_t x X_j)). aggregator: mean
        f_s = np.zeros(n, dtype=np.float)
        for i in range(n):  # each node
            if i % 1000 == 0:
                print("dealing with i: ", i)
            hyperedge_neighbors_i = search_neighbor_hyperedge(i, hyperedge_index)
            if len(hyperedge_neighbors_i) == 0:
                continue
            for ei in hyperedge_neighbors_i:  # every hyperedge
                neighbors_ei = hyperedge_neighbors_i[ei]

                # not include itself
                neighbors_idx_self = np.argwhere(neighbors_ei == i)
                neighbors_ei = np.delete(neighbors_ei, neighbors_idx_self)  # remove node itself

                edge_size = len(neighbors_ei)
                if edge_size <= 0:
                    continue
                if agg_type == 'mean':
                    f_s_i = np.mean(np.multiply(treatments[neighbors_ei], ites[neighbors_ei].reshape(-1)))  # (n x 1) * (n x 1), mean over neighbors
                    f_s_i *= agg_value
                elif agg_type == 'fix':
                    f_s_i = np.sum(np.multiply(treatments[neighbors_ei], ites[neighbors_ei].reshape(-1)))
                    f_s_i = f_s_i * edge_size / agg_value
                f_s_i = non_linear(f_s_i, nonlinear_type)
                f_s[i] += f_s_i

            num_ei = len(hyperedge_neighbors_i)
            if norm_type == 'size':
                f_s[i] /= num_ei  # normalize with hyperedge num
            elif norm_type == 'fix':
                f_s[i] /= norm_value
            elif norm_type == 'size_log':
                f_s[i] /= (1 + math.log(num_ei))
            elif norm_type == 'pow':
                f_s[i] /= (math.pow(num_ei, norm_pow))
            elif norm_type == 'try':
                f_s[i] /= ( 10 / math.pow(num_ei, 2))

        # observed y
        noise = np.random.normal(0, 1, size=n)
        w_noise = 20.0
        y = y_C * (f_y00 + alpha * f_t + beta * f_s + w_noise * noise * treatments)  # n
        y_0 = y_C * (f_y00 + 0 + beta * f_s)
        y_1 = y_C * (f_y00 + alpha * ites + beta * f_s + w_noise * noise)

        y = y * (1.0/(1+alpha+beta))
        y_0 = y_0 * (1.0/(1+alpha+beta))
        y_1 = y_1 * (1.0/(1+alpha+beta))

        y_0 = y_0.reshape(1, -1)
        y_1 = y_1.reshape(1, -1)
        Y_true = np.concatenate([y_0, y_1], axis=0)

        print('noise:', np.mean(w_noise * noise), np.std(w_noise * noise))

    elif type == 'quadratic':
        y_C = 0.03
        agg_value = 10.0

        # 1. f_y00
        W0 = np.random.normal(loc=0.0, scale=1.0, size=dim_size)
        f_y00 = np.dot(features, W0)  # f_{𝑦, 0}(x_𝑖) = W0 x_i
        # 2. ITE: f_t = t(X \delta W X^T)
        c_ft = 5
        W_t = np.random.normal(loc=0.5, scale=3.0, size=(dim_size, dim_size))
        ites = np.diag(np.dot(np.dot(features, W_t), features.T)).reshape(
            -1) + c_ft  # diag((n x d) x (d x d) x (d x n)) -> n
        f_t = np.multiply(treatments, ites)  # n

        # 3. Spillover effect: s = aggregate(t_j * \tau(x_j)) = aggregate(t_j * (W_t x X_j)). aggregator: mean
        f_s = np.zeros(n, dtype=np.float)
        for i in range(n):  # each node
            if i % 10000 == 0:
                print("dealing with i: ", i)
            hyperedge_neighbors_i = search_neighbor_hyperedge(i, hyperedge_index)  # {hyperedge_id: nodes}
            if len(hyperedge_neighbors_i) == 0:
                continue
            for ei in hyperedge_neighbors_i:  # every hyperedge
                neighbors_ei = hyperedge_neighbors_i[ei]

                # not include itself
                neighbors_idx_self = np.argwhere(neighbors_ei == i)
                neighbors_ei = np.delete(neighbors_ei, neighbors_idx_self)  # remove node itself

                edge_size = len(neighbors_ei)
                if edge_size <= 0:
                    print(1)
                    continue

                masked_features = treatments[neighbors_ei].reshape(-1,1) * features[neighbors_ei]  # n x d
                f_s_i = np.matmul(np.matmul(masked_features, W_t), masked_features.T)  # (n x d) x (d x d) x (d x n) -> n x n
                f_s_i = np.mean(f_s_i)
                f_s_i = non_linear(f_s_i, nonlinear_type)  # activate

                f_s_i *= agg_value
                f_s[i] += f_s_i

            num_ei = len(hyperedge_neighbors_i)
            if norm_type == 'size':
                f_s[i] /= num_ei  # normalize with hyperedge num
            elif norm_type == 'fix':
                f_s[i] /= norm_value
            elif norm_type == 'size_log':
                f_s[i] /= (1 + math.log(num_ei))
            elif norm_type == 'pow':
                f_s[i] /= (math.pow(num_ei, norm_pow))

        # observed y
        noise = np.random.normal(0, 1, size=n)
        w_noise = 20.0  # 2.0
        y = y_C * (f_y00 + alpha * f_t + beta * f_s) + w_noise * noise * treatments  # n
        y_0 = y_C * (f_y00 + 0 + beta * f_s)
        y_1 = y_C * (f_y00 + alpha * ites + beta * f_s) + w_noise * noise

        y_0 = y_0.reshape(1, -1)
        y_1 = y_1.reshape(1, -1)
        Y_true = np.concatenate([y_0, y_1], axis=0)

        print('noise:', np.mean(w_noise * noise), np.std(w_noise * noise))

    simulate_outcome_results = {'outcomes': y, 'Y_true': Y_true}
    return simulate_outcome_results


def simulate_goodreads(type='linear', alpha=1.0, beta=1.0, path_save=None, nonlinear_type='raw'):
    with open('../data/goodreads_processed.pickle', 'rb') as f:
        data_processed = pickle.load(f)
    features, treatment_binary, hyperedge_index = data_processed['features'], data_processed['treatment'], data_processed['hyper_index']

    feat_noise = np.random.normal(0, 0.05, features.shape)
    features += feat_noise

    # simulation begins
    simulate_outcome_results = simulate_outcome_2(features, hyperedge_index, treatment_binary, type, alpha, beta, nonlinear_type)

    simulation_data = {
        'parameter': {'alpha': alpha, 'beta': beta, 'type': type, 'nonlinear_type': nonlinear_type},
        'features': features,
        'treatments': treatment_binary,
        'outcomes': simulate_outcome_results['outcomes'],  # observed
        'Y_true': simulate_outcome_results['Y_true'],  # potential
        'hyperedge_index': hyperedge_index
    }
    if path_save is None:
        if type == 'observed':
            path_save = '../data/Simulation/GR/GoodReads_obsY.mat'
        else:
            path_save = '../data/Simulation/GR/GoodReads_sim_' + type + '_alpha' + str(alpha) + '_beta' + str(
                beta) + '.mat'
    save_flag = True
    if save_flag:
        sio.savemat(path_save, simulation_data)
        print('Data saved! Path: ', path_save)
        if type != 'observed':
            print('type=', type, ' alpha=', alpha, ' beta=',beta)
    return simulation_data

def agg_features(features, hyperedge_index, alpha=0.5):
    features_new = features.copy()
    num_of_neighbors = []
    for i in range(features.shape[0]):
        hyperedge_neighbors_i = search_neighbor_hyperedge(i, hyperedge_index)  # {hyperedge_id: nodes}
        if len(hyperedge_neighbors_i) == 0:
            continue
        feature_agg_ei_all = 0.0
        for ei in hyperedge_neighbors_i:  # every hyperedge
            neighbors_ei = hyperedge_neighbors_i[ei]

            # not include itself
            neighbors_idx_self = np.argwhere(neighbors_ei == i)
            neighbors_ei = np.delete(neighbors_ei, neighbors_idx_self)  # remove node itself

            feature_agg_ei = np.mean(features[neighbors_ei], axis=0)  # mean
            feature_agg_ei_all = feature_agg_ei_all + feature_agg_ei
        feature_agg_ei_all /= len(hyperedge_neighbors_i)

        features_new[i] = (1 - alpha) * features[i] + alpha * feature_agg_ei_all

    return features_new

def simulate_contact(type='linear', alpha=1.0, beta=1.0, path_save=None, nonlinear_type='raw'):
    # load hypergraph
    with open('../data/contact_hypergraph.pickle', 'rb') as f:
        data_processed = pickle.load(f)
    hyperedge_index = data_processed['hyper_index']
    n = np.max(hyperedge_index[0]) + 1

    # simulate features
    d_x = 50
    features = np.random.normal(loc=0.2, scale=1, size=(n, d_x))  # 10
    print('feature std: ', np.mean(np.std(features, axis=0)))

    # simulate treatments
    W = np.random.normal(0,1,size=(d_x,1))
    treatment_orin = sigmoid(0.01 * np.matmul(features, W))  # N X 1
    treatment_orin = treatment_orin.reshape(-1)
    treated_ratio = 0.49
    thresh_t = np.sort(treatment_orin)[::-1][int(treated_ratio * len(treatment_orin))]
    treatment = np.zeros_like(treatment_orin)
    treatment[np.where(treatment_orin >= thresh_t)] = 1.0
    treatment[np.where(treatment_orin < thresh_t)] = 0.0
    treated_ratio = float(np.count_nonzero(treatment)) / n
    print('treatment ratio: ', treated_ratio)

    # simulate outcomes
    simulate_outcome_results = simulate_outcome_2(features, hyperedge_index, treatment, type, alpha, beta, nonlinear_type)

    simulation_data = {
        'parameter': {'alpha':alpha, 'beta': beta, 'type': type, 'nonlinear_type': nonlinear_type},
        'features': features,
        'treatments': treatment,
        'outcomes': simulate_outcome_results['outcomes'],
        'Y_true': simulate_outcome_results['Y_true'],
        'hyperedge_index': hyperedge_index
    }
    if path_save is None:
        path_save = '../data/Simulation/contact/contact_sim_' + type + '_alpha' + str(alpha) + '_beta' + str(
                beta) + '.mat'
    save_flag = True
    if save_flag:
        sio.savemat(path_save, simulation_data)
        print('Data saved! Path: ', path_save)
        print('type=', type, ' alpha=', alpha, ' beta=', beta)
    return simulation_data

if __name__ == '__main__':
    dataset = 'contact'  # Microsoft, GoodReads, contact

    if dataset == 'GoodReads':
        type = 'linear'  # linear, quadratic
        nonlinear_type = 'raw'
        alpha = 1.0
        beta = 1.0
        path_save = '../data/Simulation/GR/GoodReads_sim_' + type + '_alpha' + str(alpha) + '_beta' + str(beta) +\
                    '_nonlinear_' + nonlinear_type + '.mat'
        simulate_goodreads(type=type, alpha=alpha, beta=beta, path_save=path_save, nonlinear_type=nonlinear_type)

    elif dataset == 'contact':
        type = 'linear'  # linear, quadratic
        nonlinear_type = 'raw'
        alpha = 1.0
        beta = 1.0
        path_save = '../data/Simulation/contact/contact_sim_' + type + '_alpha' + str(alpha) + '_beta' + str(
             beta) + '_nonlinear_' + nonlinear_type + '.mat'
        simulate_contact(type=type, alpha=alpha, beta=beta, path_save=path_save, nonlinear_type=nonlinear_type)