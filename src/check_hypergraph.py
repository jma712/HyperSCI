import numpy as np
import scipy.io as sio

def hypergraph_stats(hyperedge_index, n):
    # hyperedge size
    unique_edge, counts_edge = np.unique(hyperedge_index[1], return_counts=True)  # edgeid, size
    ave_hyperedge_size = np.mean(counts_edge)
    max_hyperedge_size = np.max(counts_edge)
    min_hyperedge_size = np.min(counts_edge)
    m = len(unique_edge)

    sz, ct = np.unique(counts_edge, return_counts=True)  # hyperedgesize, count
    counts_edge_2 = ct[np.where(sz==2)][0]

    # node degree
    unique_node, counts_node = np.unique(hyperedge_index[0], return_counts=True)  # nodeid, degree
    ave_degree = np.mean(counts_node)
    max_degree = np.max(counts_node)
    min_degree = np.min(counts_node)
    statistics = {'n': n, 'm': m, 'm>2': m-counts_edge_2,
                  'average_hyperedge_size': ave_hyperedge_size, 'min_hyperedge_size': min_hyperedge_size, 'max_hyperedge_size': max_hyperedge_size,
                  'average_degree': ave_degree, 'max_degree': max_degree, 'min_degree': min_degree}
    return statistics

if __name__ == '__main__':
    path = '/data/Simulation/MS/Microsoft_sim_quadratic_alpha1.0_beta1.0_node.mat'
    data = sio.loadmat(path)
    features, hyperedge_index = data['features'], data['hyperedge_index']

    print('loaded data from ', path)
    print(hypergraph_stats(hyperedge_index, features.shape[0]))
