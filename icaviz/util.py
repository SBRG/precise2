import re, subprocess, os
import numpy as np
import pandas as pd
from scipy import stats
from bs4 import BeautifulSoup
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from icaviz.load import *
from icaviz.util import *

import warnings

###########################
## Auto-Cluster Metadata ##
###########################

_descriptors = ['strain_description','growth_stage','temperature',
               'base_media','carbon_source_gL','nitrogen_source_gL',
               'electron_acceptor','supplement','environment',
               'taxonomy_id']

def decision_tree_helper(k,features,label_encoder,
                         min_samples_leaf,max_leaf_nodes,ica_data):
    
    # Run one-hot encoding on labels
    onehot = OneHotEncoder()
    final_features = onehot.fit_transform(features)
    
    # Initialize output dataframe
    DF_mode = pd.DataFrame(ica_data.A.loc[k].sort_values())
    DF_mode['exp_id'] = DF_mode.index
    DF_mode.index = [ica_data.exp2num[name] for name in DF_mode.index]
    
    # Run Decision Tree Regressor
    clf = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf,
                                criterion='mae',max_leaf_nodes=max_leaf_nodes)
    clf.fit(final_features,ica_data.A.loc[k])
    
    # Load tree information
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature

    group_lev = 0
    stack = [(0,-1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        # Add feature split to dataframe
        feat = feature[node_id]
        # Check for leaf
        if feat != -2:
            group_lev += 1
            cat = np.where(onehot.feature_indices_<=feat)[0][-1]
            index = features[features.iloc[:,cat] == \
                             feat-onehot.feature_indices_[cat]].index
            col = features.columns[cat]
            
            with warnings.catch_warnings(): # TODO: Remove this when Deprecation warning no longer shows up
                warnings.filterwarnings("ignore",category=DeprecationWarning)
                val = label_encoder[col].inverse_transform(feat-onehot.\
                                                           feature_indices_[cat])
            
            # Write category name if it is a small category
            if len(index) < len(DF_mode)/2:
                DF_mode.loc[index,'group_'+str(group_lev)] = col + ': ' + \
                                                             str(val)
           
            # Else write opposite of category name
            else:
                index = set(DF_mode.index)-set(index)
                if len(ica_data.metadata[col].unique()) == 2:
                    val = list(set(ica_data.metadata[col].unique()) - set([val]))[0]
                    DF_mode.loc[index,'group_'+str(group_lev)] = col + ': ' + \
                                                                 str(val)
                else:
                    DF_mode.loc[index,'group_'+str(group_lev)] = col + \
                                                                 ': NOT ' + \
                                                                 str(val)
                
        # Check if node has children
        if children_left[node_id] != -1:
            stack.append((children_left[node_id], parent_depth + 1))
        if children_right[node_id] != -1:
            stack.append((children_right[node_id], parent_depth + 1))

    # Create full group dataframe
    groups = DF_mode[[col for col in DF_mode.columns 
                      if str(col).startswith('group_')]]
    DF_mode['group'] = groups.fillna('').apply(lambda x: \
                            re.sub('\n+','\n','\n'.join(x)), axis=1)
    DF_mode = DF_mode[[k,'exp_id','group']]
    
    # Fix empty carbon/nitrogen/electron fields
    DF_mode['group'] = [re.sub('carbon-source:( \n|$)','base-media: LB\n',x) \
                        for x in DF_mode['group']]
    DF_mode['group'] = [re.sub('nitrogen-source:( \n|$)','base-media: LB\n',x) \
                        for x in DF_mode['group']]
    DF_mode['group'] = [re.sub('electron-acceptor:( \n|$)','anaerobic\n',x) \
                        for x in DF_mode['group']]
    DF_mode['group'] = ['Other' if x.strip() == '' else x 
                        for x in DF_mode['group']]
    DF_mode['group'] = [x.strip() for x in DF_mode['group']]

    return DF_mode

def cluster_metadata(ica_data,k,min_samples_leaf=2,max_groups=10):
    
    max_leaf_nodes=max_groups

    # Turn string labels for metadata into integers
    DF_features = pd.DataFrame(index=ica_data.metadata.index)
    label_encoder = {}
    for desc in _descriptors:
        label_encoder[desc] = LabelEncoder()
        DF_features[desc] = label_encoder[desc].fit_transform(ica_data.metadata[desc])
        
    # Use decision tree to cluster conditions and output to dataframe
    DF_mode = decision_tree_helper(k,DF_features,label_encoder,
                                   min_samples_leaf,max_leaf_nodes,ica_data)

    # Make sure that there are only n categories
    while len(DF_mode.group.unique()) > max_groups:
        max_leaf_nodes -= 1
        DF_mode = decision_tree_helper(k,DF_features,label_encoder,
                                       min_samples_leaf,max_leaf_nodes,ica_data)

    return DF_mode
    

###################
## MOTIF FINDING ##
###################

def find_motifs(ica_data,k,palindrome=False,nmotifs=5,upstream=500,downstream=100,
                verbose=True,force=False,evt=0.001,maxw=40):
    
    if not os.path.isdir('motifs'):
        os.mkdir('motifs')
    
    # Get list of operons in component
    enriched_genes = ica_data.show_enriched(k).index
    enriched_operons = ica_data.gene_info.loc[enriched_genes]
    n_operons = len(enriched_operons.operon.unique())
    
    # Return empty dataframe if under 4 operons or over 200 operons exist
    if n_operons <= 4 or n_operons > 200:
        return pd.DataFrame(columns = ['motif_frac']),pd.DataFrame()
    
    # Get upstream sequences
    list2struct = []
    seqs = []
    for name,group in enriched_operons.groupby('operon'):
        genes = ','.join(group.gene_name)
        ids = ','.join(group.index)
        if all(group.strand == '+'):
            pos = min(group.start)
            start_pos = max(0,pos-upstream)
            seq = ica_data.fasta[start_pos:pos+downstream]
            seq.id = name
            list2struct.append([name,genes,ids,
                                start_pos,'+',str(seq.seq)])
            seqs.append(seq)
        elif all(group.strand == '-'):
            pos = max(group.stop)
            start_pos = max(0,pos-downstream)
            seq = ica_data.fasta[start_pos:pos+upstream]
            seq.id = name
            list2struct.append([name,genes,ids,
                                start_pos,'-',str(seq.seq)])
            seqs.append(seq)
        else:
            raise ValueError('Operon contains genes on both strands:',name)
            
    DF_seqs = pd.DataFrame(list2struct,columns=['operon','genes','locus_tags','start_pos','strand','seq']).set_index('operon')

    # Add TRN info
    tf_col = []
    for genes in DF_seqs.locus_tags:
        tfs = []
        for gene in genes.split(','):
            tfs += ica_data.trn[ica_data.trn.gene_id == gene].TF.unique().tolist()
        tf_col.append(','.join(list(set(tfs))))
    DF_seqs.loc[:,'TFs'] = tf_col

    # Run MEME
    if verbose:
        print('Finding motifs for {:d} sequences'.format(len(seqs)))
    if palindrome:
        comp_dir = 'motifs/' + re.sub('/','_','{}_pal'.format(k))
    else:
        comp_dir = 'motifs/' + re.sub('/','_',str(k))
    
    # Skip intensive tasks on rerun
    if force or not os.path.isdir(comp_dir):
    
        # Write sequence to file
        fasta = 'motifs/' + re.sub('/','_','{}.fasta'.format(k))
        SeqIO.write(seqs,fasta,'fasta')

        # Minimum number of total sites to find
        minsites = max(2,int(n_operons/3)) 
        
        cmd = ['meme',fasta,'-oc',comp_dir,
               '-dna','-mod','zoops','-p','8','-nmotifs',str(nmotifs),
               '-evt',str(evt),'-minw','6','-maxw',str(maxw),'-allw',
               '-minsites',str(minsites)]
        if palindrome:
            cmd.append('-pal')
        subprocess.call(cmd)

    # Save results
    result = parse_meme_output(comp_dir,DF_seqs,verbose=verbose,evt=evt)
    ica_data.motif_info[k] = result
    return result

def parse_meme_output(directory,DF_seqs,verbose=True,evt=0.001):

    # Read MEME results
    with open(directory+'/meme.xml','r') as f:
        result_file = BeautifulSoup(f.read(),'lxml')

    # Convert to motif XML file to dataframes: (overall,[individual_motif])
    DF_overall = pd.DataFrame(columns=['e_value','sites','width','consensus'])
    dfs = []
    for motif in result_file.find_all('motif'):

        # Motif statistics
        DF_overall.loc[motif['id'],'e_value'] = np.float64(motif['e_value'])
        DF_overall.loc[motif['id'],'sites']  = motif['sites']
        DF_overall.loc[motif['id'],'width']  = motif['width']
        DF_overall.loc[motif['id'],'consensus']  = motif['name']
        DF_overall.loc[motif['id'],'motif_name'] = motif['alt']

        # Map Sequence to name

        list_to_struct = []
        for seq in result_file.find_all('sequence'):
            list_to_struct.append([seq['id'],seq['name']])
        df_names = pd.DataFrame(list_to_struct,columns=['seq_id','operon'])

        # Get motif sites

        list_to_struct = []
        for site in motif.find_all('contributing_site'):

            site_seq = ''.join([letter['letter_id'] 
                                for letter in site.find_all('letter_ref')
                               ])
            data = [site['position'],site['pvalue'],site['sequence_id'],
                    site.left_flank.string,site_seq,site.right_flank.string]
            list_to_struct.append(data)
            
        tmp_df = pd.DataFrame(list_to_struct,columns=['rel_position','pvalue','seq_id',
                                                      'left_flank','site_seq','right_flank'])  

        # Combine motif sites with sequence to name mapper
        DF_meme = pd.merge(tmp_df,df_names)
        DF_meme = DF_meme.set_index('operon').sort_index().drop('seq_id',axis=1)
        DF_meme = pd.concat([DF_meme,DF_seqs],axis=1,sort=True)
        DF_meme.index.name = motif['id']
        
        # Report number of sequences with motif
        DF_overall.loc[motif['id'],'motif_frac'] = np.true_divide(sum(DF_meme.rel_position.notnull()),len(DF_meme))
        dfs.append(DF_meme)
        
    if len(dfs) == 0:
        if verbose:
            print('No motif found with E-value < {0:.1e}'.format(evt))
        return pd.DataFrame(columns=['e_value','sites','width','consensus','motif_frac']),[]
       
    return DF_overall,pd.concat({df.index.name:df for df in dfs})

def compare_motifs(k,motif_db,force=False,evt=.001):
    motif_file = 'motifs/' + re.sub('/','_',str(k)) + '/meme.txt'
    out_dir = 'motifs/' + re.sub('/','_',str(k))+ '/tomtom_out/'
    if not os.path.isdir(out_dir) or force:
        subprocess.call(['tomtom','-oc',out_dir,'-thresh',str(evt),'-incomplete-scores','-png',motif_file,motif_db])
    DF_tomtom = pd.read_csv(os.path.join(out_dir,'tomtom.tsv'),sep='\t',skipfooter=3,engine='python')
    
    if len(DF_tomtom) > 0:
        row = DF_tomtom.iloc[0]
        print(row['Target_ID'])
        tf_name = row['Target_ID'][:4].strip('_')
        lines = 'Motif similar to {} (E-value: {:.2e})'.format(tf_name,row['E-value'])
        files = out_dir+'align_'+row['Query_ID']+'_0_-'+row['Target_ID']+'.png'
        if not os.path.isfile(files):
            files = out_dir+'align_'+row['Query_ID']+'_0_+'+row['Target_ID']+'.png'
        with open(out_dir+'/tomtom.xml','r') as f:
            result_file = BeautifulSoup(f.read(),'lxml')
        motif_names = [motif['alt'] for motif in result_file.find('queries').find_all('motif')]
        idx = int(result_file.find('matches').query['idx'])
        
        return motif_names[idx],lines,files
    else:
        return -1,'',''

#############################
## Differential Activities ##
#############################

def diff_act(ica_data,s1_list,s2_list,fdr_rate=.1,lfc=5):
    res = pd.DataFrame(index=ica_data.names)
    for k in res.index:
        a1 = ica_data.A.loc[k,s1_list].mean()
        a2 = ica_data.A.loc[k,s2_list].mean()
        res.loc[k,'LFC'] = a2-a1
        res.loc[k,'pvalue'] = 1-ica_data.dist[k](abs(a1-a2))
    final = FDR(res,fdr_rate)
    return final[(abs(final.LFC) > lfc)].sort_values('LFC',ascending=False)
    
###################################
## Cumulative Explained Variance ##
###################################

def plot_rec_var(ica_data,ref_conds=None,genes=None,samples=None,modulons=None,plot=True):
    
    # This uses the formula of Cumulative Explained Variance that is described in Sastry et al., Nat. Comm., 2019
    # For this to be accurate, you must include reference conditions if you did not load ica_data with the centered X
    # For PRECISE, ref_conds are ['baseline__wt_glc__1','baseline__wt_glc__2']
    
    # Check inputs
    if genes is None:
        genes = ica_data.X.index
    elif isinstance(genes,str):
        genes = [genes]
    if samples is None:
        samples = ica_data.X.columns
    elif isinstance(samples,str):
        samples = [samples]
    if modulons is None:
        modulons = ica_data.S.columns
    elif isinstance(modulons,str):
        modulons = [modulons]

    # Account for normalization procedures before ICA (X=SA-x_mean)
    if ref_conds is None:
        X = ica_data.X
    else:
        X = ica_data.X.subtract(ica_data.X[ref_conds].mean(axis=1),axis=0)
    
    baseline = pd.DataFrame(np.subtract(X,X.values.mean(axis=0,keepdims=True)),
                    index=ica_data.S.index,columns=ica_data.A.columns)
    baseline = baseline.loc[genes]
    
    # Initialize variables
    base_err = np.linalg.norm(baseline)**2
    SA = np.zeros(baseline.shape)
    rec_var = [0]
    sa_arrs = {}
    sa_weights = {}
    
    # Get individual modulon contributions
    for k in modulons:
        sa_arr = np.dot(ica_data.S.loc[genes,k].values.reshape(len(genes),1),
                         ica_data.A.loc[k,samples].values.reshape(1,len(samples)))
        sa_arrs[k] = sa_arr
        sa_weights[k] = np.sum(sa_arr**2)
    
    # Sum components in order of most important component first
    sorted_mods = sorted(sa_weights,key=sa_weights.get,reverse=True)
    # Compute reconstructed variance
    for k in sorted_mods:
        SA = SA + sa_arrs[k]
        sa_err = np.linalg.norm(SA-baseline)**2
        rec_var.append((1-sa_err/base_err)*100)

    if plot:
        fig,ax = plt.subplots()
        ax.plot(rec_var)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained Variance')
        ax.set_ylim([0,100])
        return ax
    else:
        return pd.Series(rec_var[1:],index=sorted_mods)
