import scipy.io as sio
import torch
import scipy.sparse as sp
import numpy as np
import random
import torch.nn.functional as F
from scipy.sparse import csc_matrix
import  scipy.sparse as sparse
import pandas as pd

device = torch.device("cuda:0")

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]

    x = x.squeeze()
    y = y.squeeze()

    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).cpu().detach()
    eff_lam = (lam / M_mean).cpu().detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        #row = row.cuda()
        #col = col.cuda()
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        #temp_term = temp_term.cuda()
        #a = a.cuda()
        #b = b.cuda()
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            #u = u.cuda()
            u = u.to(device)
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        #v = v.cuda()
        v = v.to(device)

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    if cuda:
        #D = D.cuda()
        D = D.to(device)

    return D, Mlam

def pdist2sq(x_t, x_cf):
    C = -2 * torch.matmul(x_t,torch.t(x_cf))
    n_t = torch.sum(x_t * x_t, 1, True)
    n_cf = torch.sum(x_cf * x_cf, 1, True)
    D = (C + torch.t(n_cf)) + n_t
    return D

def mmd2_rbf(Xt, Xc, p,sig):
    """ Computes the l2-RBF MMD for X given t """

    Kcc = torch.exp(-pdist2sq(Xc,Xc)/(sig)**2)
    Kct = torch.exp(-pdist2sq(Xc,Xt)/(sig)**2)
    Ktt = torch.exp(-pdist2sq(Xt,Xt)/(sig)**2)

    m = Xc.shape[0]
    n = Xt.shape[0]

    mmd = (1.0-p)**2/(m*(m-1.0))*(torch.sum(Kcc)-m)
    mmd = mmd + (p) ** 2/(n*(n-1.0))*(torch.sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*torch.sum(Kct)
    mmd = 4.0*mmd

    return mmd

def mmd2_lin(Xt, Xc,p):
    ''' Linear MMD '''
    mean_control = torch.mean(Xc,0)
    mean_treated = torch.mean(Xt,0)

    mmd = torch.sum((2.0*p*mean_treated - 2.0*(1.0-p)*mean_control) ** 2)

    return mmd

def safe_sqrt(x, lbound=1e-10):
    ''' Numerically safe version of pytorch sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, np.inf))

def get_hyperedge_attr(features, hyperedge_index, type='mean'):
    # input: features: tensor N x F; hyperedge_index: 2 x |sum of all hyperedge size|
    # return hyperedge_attr: tensor, M x F
    #features = torch.FloatTensor([[0, 0.1, 0.2], [1.1, 1.2, 1.3], [2., 2.1, 2.2], [3.1,3.2,3.3], [4, 4.1, 4.2], [5,5,5]])
    #hyperedge_index = torch.LongTensor([[0,1,0,3,4,5,1],[0,0,1,1,1,2,2]])
    if type == 'mean':
        # hyperedge_attr = features[hyperedge_index[0]]  # |sum of all hyperedge size| x F
        # index_start =  # M, the start index of every hyperedge
        # hyperedge_attr = torch.tensor_split(hyperedge_attr, index_start)  #
        hyperedge_attr = None
        samples = features[hyperedge_index[0]]
        labels = hyperedge_index[1]

        labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

        hyperedge_attr = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
        hyperedge_attr = hyperedge_attr / labels_count.float().unsqueeze(1)
    return hyperedge_attr

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def project_hypergraph(n, hyperedge_index, type='hyper_index'):
    # inner product:  uid, tid + uid, tid => uid - tid - uid
    # hyperedge_index = torch.LongTensor([[0], [0]])
    # hyperedge_index = sparse.eye(n, dtype=np.int8)
    # return hyperedge_index

    df = pd.DataFrame(data={'uid': hyperedge_index[0], 'tid': hyperedge_index[1]})
    df = pd.merge(df, df, on='tid')

    df_team_num = df[df['uid_x'] < df['uid_y']]   # u<v
    df_team_num = df_team_num.groupby(['uid_x', 'uid_y'])['tid'].count().reset_index(
        name='Count').sort_values(['Count'], ascending=False)
    df_team_num = df_team_num[df_team_num['Count'] > 1]
    print('num of high-order repeat: ', df_team_num.shape[0], ' highest', df_team_num['Count'].max())

    df = df.loc[:, ['uid_x', 'uid_y']].drop_duplicates()  # (uid, uid) with no repeat,  (u, v) and (v, u) are both in

    if type == 'graph_index':
        df_self = pd.DataFrame(data={'uid_x': np.arange(n), 'uid_y': np.arange(n)})
        df = df.append(df_self, ignore_index=True).drop_duplicates()
        df = df.sort_values(by=['uid_x', 'uid_y'], ascending=True)  # add edge (i,i)

        edge_num = df.shape[0]
        rows = df.loc[:, 'uid_x'].values.reshape(-1)
        cols  = df.loc[:, 'uid_y'].values.reshape(-1)

        data = np.ones(edge_num)
        adj_sparse = csc_matrix((data, (rows, cols)), shape=(n, n))
        projected_graph = adj_sparse

        print('projected the hypergraph into a plain graph with edge num: ', (edge_num-n)/2)

    elif type == 'hyper_index':
        df = df.drop(df[df.uid_x == df.uid_y].index)  # remove self loop
        df = df.drop(df[df.uid_x >= df.uid_y].index)  # just keep (u, v), u<v
        df = df.sort_values(by=['uid_x', 'uid_y'], ascending=True)
        edge_num = df.shape[0]
        df.insert(loc=len(df.columns), column='tid', value=range(edge_num))

        df_b = df.loc[:, ['uid_y', 'tid']].rename(columns={'uid_y': 'uid_x'})
        df = df.loc[:, ['uid_x', 'tid']].append(df_b)

        df = df.sort_values(by=['tid', 'uid_x'])
        hyperedge_index = df.values.T
        projected_graph = hyperedge_index
        print('projected the hypergraph into a plain graph with edge num: ', edge_num)

    return projected_graph

if __name__ == '__main__':
    hyperedge_index = np.array([[0, 3, 0, 4, 0, 3, 2],[0, 0, 1, 1, 2, 2, 2]])
    project_hypergraph(5, hyperedge_index, 'hyper_index')
