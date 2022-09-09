import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from sklearn import metrics
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, GATConv
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HyperSCI(nn.Module):
    def __init__(self, args, x_dim):
        super(HyperSCI, self).__init__()

        self.h_dim = args.h_dim
        self.n_out = args.n_out
        self.g_dim = args.g_dim
        self.dropout = args.dropout
        self.graph_model = args.graph_model
        self.encoder = args.encoder

        self.phi_x = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device))

        if self.encoder == 'gat':
            self.hgnn = gnn.HypergraphConv(self.h_dim, self.g_dim, use_attention=True, heads=2, concat=False).to(device)
        elif self.encoder == 'gcn':
            self.hgnn = gnn.HypergraphConv(self.h_dim, self.g_dim).to(device)
        else:
            self.hgnn = nn.Sequential(nn.Linear(self.h_dim, self.g_dim).to(device))

        # potential outcome
        self.y_pred_dim = self.h_dim + self.g_dim
        self.out_t00 = [nn.Linear(self.y_pred_dim, self.y_pred_dim).to(device) for i in range(self.n_out)]
        self.out_t10 = [nn.Linear(self.y_pred_dim, self.y_pred_dim).to(device) for i in range(self.n_out)]
        self.out_t01 = nn.Linear(self.y_pred_dim, 1).to(device)
        self.out_t11 = nn.Linear(self.y_pred_dim, 1).to(device)

    def forward(self, features, treatments, hyperedge_index, weight_g=1.0):
        phi_x = self.phi_x(features)
        phi_x_t = torch.mul(treatments.view(-1,1), phi_x)

        hyperedge_attr = None
        if self.encoder == 'gat':
            hyperedge_attr = utils.get_hyperedge_attr(phi_x_t, hyperedge_index, type='mean')
        if self.encoder == 'gat' or self.encoder == 'gcn':
            rep_hgnn = self.hgnn(x=phi_x_t, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)  # hypergnn

        rep_hgnn = F.dropout(rep_hgnn, self.dropout, training=self.training)
        rep_post = torch.cat([phi_x, rep_hgnn], dim=1)

        # potential outcome
        if self.n_out == 0:
            y00 = rep_post
            y10 = rep_post
        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep_post))
            y10 = F.relu(self.out_t10[i](rep_post))

        y0_pred = self.out_t01(y00).view(-1)
        y1_pred = self.out_t11(y10).view(-1)

        results = {
            'y1_pred':  y1_pred,  'y0_pred': y0_pred, 'rep': phi_x
        }

        return results

class GraphSCI(nn.Module):
    def __init__(self, args, x_dim):
        super(GraphSCI, self).__init__()

        self.h_dim = args.h_dim
        self.n_out = args.n_out
        self.g_dim = args.g_dim
        self.dropout = args.dropout
        self.encoder = args.encoder

        self.phi_x = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device))

        self.gnn = GCNConv(self.h_dim, self.g_dim).to(device)

        self.y_rep_dim = self.h_dim + self.g_dim

        # potential outcome
        self.out_t00 = [nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device) for i in range(self.n_out)]
        self.out_t10 = [nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device) for i in range(self.n_out)]
        self.out_t01 = nn.Linear(self.y_rep_dim, 1).to(device)
        self.out_t11 = nn.Linear(self.y_rep_dim, 1).to(device)

    def forward(self, features, treatments, edge_index):
        phi_x = self.phi_x(features)
        phi_x_t = torch.mul(treatments.view(-1, 1), phi_x)

        rep_gnn = self.gnn(x=phi_x_t, edge_index=edge_index)

        rep_gnn = F.dropout(rep_gnn, self.dropout, training=self.training)
        rep_post = torch.cat([phi_x, rep_gnn], dim=1)

        # potential outcome
        if self.n_out == 0:
            y00 = rep_post
            y10 = rep_post
        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep_post))
            y10 = F.relu(self.out_t10[i](rep_post))

        y0_pred = self.out_t01(y00).view(-1)
        y1_pred = self.out_t11(y10).view(-1)

        results = {
            'y1_pred': y1_pred, 'y0_pred': y0_pred, 'rep': phi_x
        }

        return results
