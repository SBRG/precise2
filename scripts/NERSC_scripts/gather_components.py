#!/bin/python

import pandas as pd
import numpy as np
import os,argparse

def main(n_runs,n_iter,input_dir):
    data_dirs = [os.path.join(input_dir,'{:d}'.format(i)) for i in range(1,n_runs+1)]

    # Load S and A matrices
    all_S = pd.concat([pd.read_csv(os.path.join(data_dir,'S.csv'),index_col=0)\
        for data_dir in data_dirs],axis=1)
    
    all_S.columns = range(len(all_S.columns))
    
    all_A = pd.concat([pd.read_csv(os.path.join(data_dir,'A.csv'),index_col=0)\
        for data_dir in data_dirs]).reset_index(drop=True)
    
    all_stats = pd.concat([pd.read_csv(os.path.join(data_dir, \
        'component_stats.csv'),index_col=0) \
        for data_dir in data_dirs]).reset_index()

    # Make sure no cluster contains more than <n_iter> components
    assert(all(all_stats['count'] <= n_iter))

    # Calculate distance matrix
    diff_mat = 1-abs(all_S.corr()).values

    # Cluster components
    comp_idx = range(len(diff_mat))
    i = comp_idx[0]
    comp_dict = {}
    while len(comp_idx) > 0:
        i = comp_idx[0]
        identical = np.where(diff_mat[i] < 0.1)[0]
        comp_dict[i] = identical
        comp_idx = sorted(set(comp_idx).difference(set(identical)))

    comp_dist = {}
    for i,lst in comp_dict.items():
        comp_dist[i] = all_stats.loc[lst,'count']

    # Get statistics about each component
    resdf = pd.DataFrame([[lst.min(),
                           lst.max(),
                           lst.mean(),
                           lst.std(),
                           len(lst)] for lst in comp_dist.values()],
                         index=comp_dist.keys(),
                         columns=['Min','Max','Mean','STD','Length'])

    # Only keep components that exist in every single run
    good_comps = resdf[resdf.Length == n_runs].index

    # Save iModulons
    all_S[good_comps].T.reset_index(drop=True).T.to_csv('S.csv')
    all_A.loc[good_comps].reset_index(drop=True).to_csv('A.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_runs',type=int,help='Number of robust ICA runs',default=10)
    parser.add_argument('n_iter',type=int,help='Number of iterations per ICA run',default=100)
    parser.add_argument('input_dir',help='Directory containing ICA runs')
    args = parser.parse_args()
    main(args.n_runs,args.n_iter,args.input_dir)
