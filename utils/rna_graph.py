import subprocess
import numpy as np
import scipy.sparse as sp
import os
import random
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import dgl
import torch
from collections import defaultdict

def fold_seq_rnaplfold(seq, w, l, cutoff, no_lonely_bps):
    np.random.seed(random.seed())
    name = str(np.random.rand())
    # Call RNAplfold on command line.
    no_lonely_bps_str = ""
    if no_lonely_bps:
        no_lonely_bps_str = "--noLP"
    cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f --id-prefix %s %s' % (seq, w, l, cutoff, name, no_lonely_bps_str)
    ret = subprocess.call(cmd, shell=True)

    # # assemble adjacency matrix
    row_col, prob = [], []
    length = len(seq)
    
    max_degree = 2
    
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            prob.append(1.)
        if i != 0:
            row_col.append((i, i - 1))
            prob.append(1.)
    #Extract base pair information.
    name += '_0001_dp.ps'
    start_flag = False
    with open(name) as f:
        for line in f:
            if start_flag:
                values = line.split()
                if len(values) == 4:
                    source_id = int(values[0]) - 1
                    dest_id = int(values[1]) - 1
                    avg_prob = float(values[2])
                    # source_id < dest_id
                    row_col.append((source_id, dest_id))
                    prob.append(avg_prob ** 2)
                    row_col.append((dest_id, source_id))
                    prob.append(avg_prob ** 2)
            if 'start of base pair probability data' in line:
                start_flag = True
                
    prob_matrix = sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length))            
    # delete RNAplfold output file.
    os.remove(name)

    # placeholder for dot-bracket structure
    return prob_matrix

def fold_rna_from_file(all_seq):

    pool = Pool(int(os.cpu_count() * 2 / 3))
    
    fold_func = partial(fold_seq_rnaplfold, w=99, l=99, cutoff=1e-4, no_lonely_bps=True)
    res = list(pool.imap(fold_func, all_seq))

    sp_prob_matrix = []
    for prob_mat in res:
        sp_prob_matrix.append(prob_mat)

    return sp_prob_matrix


def _constructGraph(seq, csr_matrix):

    src, dst = csr_matrix.nonzero()
    edge_weights = csr_matrix[src, dst].astype(np.float32)
    edge_weights = edge_weights.T

    # Create the graph
    grh = dgl.graph((src, dst), num_nodes=99)
        
    grh.edata['feat'] = torch.tensor(edge_weights, dtype=torch.float32)

    return grh


def convert2dglgraph(seq_list):
    csr_matrixs = fold_rna_from_file(seq_list)
    dgl_graph_list = []
    for i in tqdm(range(len(csr_matrixs))):
        dgl_graph_list.append(_constructGraph(seq_list[i], csr_matrixs[i]))
    return dgl_graph_list


