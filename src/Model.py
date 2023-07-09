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
        self.skip_type = args.skip
        self.num_gnn_layer = args.num_gnn_layer
        self.phi_layer = args.phi_layer
        self.activate = args.activate

        self.phi_x = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device)) if self.phi_layer == 1 else \
            nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device), nn.Linear(self.h_dim, self.h_dim).to(device))
        self.phi_x_p = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device), nn.Linear(self.h_dim, self.h_dim).to(device))

        if self.encoder == 'gat':
            self.hgnn = gnn.HypergraphConv(self.h_dim, self.g_dim, use_attention=True, heads=2, concat=False, dropout=0.5).to(device)
            self.hgnn_more = [gnn.HypergraphConv(self.g_dim, self.g_dim, use_attention=True, heads=2, concat=False, dropout=0.5).to(device) for i in range(self.num_gnn_layer-1)]
        elif self.encoder == 'gcn':
            self.hgnn = gnn.HypergraphConv(self.h_dim, self.g_dim).to(device)
            self.hgnn_more = [gnn.HypergraphConv(self.g_dim, self.g_dim).to(device) for i in range(self.num_gnn_layer-1)]
        else:
            self.hgnn = nn.Sequential(nn.Linear(self.h_dim, self.g_dim).to(device))  # MLP
            self.hgnn_more = [nn.Sequential(nn.Linear(self.g_dim, self.g_dim).to(device)) for i in range(self.num_gnn_layer-1)]

        if self.skip_type == '123':
            self.y_rep_dim = x_dim + self.h_dim + self.g_dim
        elif self.skip_type == '23':  # phi_x + g_rep
            self.y_rep_dim = self.h_dim + self.g_dim

        # prediction for potential outcome
        self.out_t00 = [nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device) for i in range(self.n_out)]
        self.out_t10 = [nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device) for i in range(self.n_out)]
        self.out_t01 = nn.Linear(self.y_rep_dim, 1).to(device)
        self.out_t11 = nn.Linear(self.y_rep_dim, 1).to(device)

    def forward(self, features, treatments, hyperedge_index):
        phi_x = self.phi_x(features)
        phi_x_t = torch.mul(treatments.view(-1,1), phi_x)
        phi_x_p = phi_x  # if use phi_x_p, set self.phi_x_p(features)

        hyperedge_attr = None
        if self.encoder == 'gat':
            hyperedge_attr = utils.get_hyperedge_attr(phi_x_t, hyperedge_index, type='mean')
        if self.encoder == 'gat' or self.encoder == 'gcn':
            rep_hgnn = self.hgnn(x=phi_x_t, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)  # hypergnn
            for i in range(self.num_gnn_layer-1):
                if self.activate:
                    rep_hgnn = F.relu(rep_hgnn)
                if self.encoder == 'gat':
                    hyperedge_attr = utils.get_hyperedge_attr(rep_hgnn, hyperedge_index, type='mean')
                rep_hgnn = self.hgnn_more[i](x=rep_hgnn, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)
        else:  # mlp
            rep_hgnn = self.hgnn(phi_x_t)
            for i in range(self.num_gnn_layer - 1):
                if self.activate:
                    rep_hgnn = F.relu(rep_hgnn)
                rep_hgnn = self.hgnn_more[i](rep_hgnn)
        if self.activate:
            rep_hgnn = F.relu(rep_hgnn)
        rep_hgnn = F.dropout(rep_hgnn, self.dropout, training=self.training)

        if self.skip_type == '123':
            rep_post_0 = torch.cat([features, torch.zeros_like(phi_x_p), rep_hgnn], dim=1)
            rep_post_1 = torch.cat([features, phi_x_p, rep_hgnn], dim=1)
        elif self.skip_type == '23':
            rep_post_0 = torch.cat([torch.zeros_like(phi_x_p), rep_hgnn], dim=1)
            rep_post_1 = torch.cat([phi_x_p, rep_hgnn], dim=1)

        # potential outcome
        if self.n_out == 0:
            y00 = rep_post_0
            y10 = rep_post_1
        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep_post_0))
            #y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep_post_1))
            #y10 = F.dropout(y10, self.dropout, training=self.training)

        y0_pred = self.out_t01(y00).view(-1)
        y1_pred = self.out_t11(y10).view(-1)

        results = {'y1_pred':  y1_pred,  'y0_pred': y0_pred, 'rep': phi_x}

        return results


class GraphSCI(nn.Module):
    def __init__(self, args, x_dim):
        super(GraphSCI, self).__init__()

        self.h_dim = args.h_dim
        self.n_out = args.n_out
        self.g_dim = args.g_dim
        self.dropout = args.dropout
        self.encoder = args.encoder
        self.skip_type = args.skip
        self.num_gnn_layer = args.num_gnn_layer
        self.phi_layer = args.phi_layer
        self.activate = args.activate

        self.phi_x = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device)) if self.phi_layer == 1 else \
            nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device),nn.Linear(self.h_dim, self.h_dim).to(device))

        self.phi_x_p = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device),
                                   nn.Linear(self.h_dim, self.h_dim).to(device))

        if self.encoder == 'gcn':
            self.gnn = GCNConv(self.h_dim, self.g_dim).to(device)
            self.gnn_more = [GCNConv(self.g_dim, self.g_dim).to(device) for i in range(self.num_gnn_layer - 1)]
        else:
            self.gnn = nn.Sequential(nn.Linear(self.h_dim, self.g_dim).to(device))  # MLP
            self.gnn_more = [nn.Sequential(nn.Linear(self.g_dim, self.g_dim).to(device)) for i in range(self.num_gnn_layer - 1)]

        if self.skip_type == '123':
            self.y_rep_dim = x_dim + self.h_dim + self.g_dim
        elif self.skip_type == '23':  # phi_x + g_rep
            self.y_rep_dim = self.h_dim + self.g_dim

        # potential outcome
        self.out_t00 = [nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device) for i in range(self.n_out)]
        self.out_t10 = [nn.Linear(self.y_rep_dim, self.y_rep_dim).to(device) for i in range(self.n_out)]
        self.out_t01 = nn.Linear(self.y_rep_dim, 1).to(device)
        self.out_t11 = nn.Linear(self.y_rep_dim, 1).to(device)

    def forward(self, features, treatments, edge_index):
        phi_x = self.phi_x(features)
        phi_x_p = phi_x  # self.phi_x_p(features)
        phi_x_t = torch.mul(treatments.view(-1, 1), phi_x)

        if self.encoder == 'gcn':
            rep_gnn = self.gnn(x=phi_x_t, edge_index=edge_index)  # hypergnn
            for i in range(self.num_gnn_layer - 1):
                if self.activate:
                    rep_gnn = F.relu(rep_gnn)
                rep_gnn = self.gnn_more[i](x=rep_gnn, edge_index=edge_index)
        else:
            rep_gnn = self.gnn(phi_x_t)
            for i in range(self.num_gnn_layer - 1):
                if self.activate:
                    rep_gnn = F.relu(rep_gnn)
                rep_gnn = self.gnn_more[i](rep_gnn)
        if self.activate:
            rep_gnn = F.relu(rep_gnn)
        rep_gnn = F.dropout(rep_gnn, self.dropout, training=self.training)
        rep_post_0 = torch.cat([torch.zeros_like(phi_x_p), rep_gnn], dim=1)
        rep_post_1 = torch.cat([phi_x_p, rep_gnn], dim=1)

        # prediction for potential outcome
        if self.n_out == 0:
            y00 = rep_post_0
            y10 = rep_post_1
        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep_post))
            #y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep_post))
            #y10 = F.dropout(y10, self.dropout, training=self.training)

        y0_pred = self.out_t01(y00).view(-1)
        y1_pred = self.out_t11(y10).view(-1)

        results = {'y1_pred': y1_pred, 'y0_pred': y0_pred, 'rep': phi_x}

        return results

