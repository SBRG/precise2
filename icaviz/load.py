import pandas as pd
import numpy as np
from scipy import stats,special
from Bio import SeqIO
import os, warnings
from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection

###########################
## TF Enrichment scripts ##
###########################

def contingency(set1,set2,all_genes):
    """Creates contingency table for gene enrichment
        set1: Set of genes (e.g. regulon)
        set2: Set of genes (e.g. i-modulon)
        all_genes: Set of all genes
    """
        
    tp = len(set1 & set2)
    fp = len(set2 - set1)
    tn = len(all_genes - set1 - set2)
    fn = len(set1 - set2)
    return [[tp,fp],[fn,tn]]
    
def FDR(p_values,fdr_rate,total=None):
    """Runs false detection correction over a pandas Dataframe
        p_values: Pandas Dataframe with 'pvalue' column
        fdr_rate: False detection rate
        total: Total number of tests (for multi-enrichment)
    """
    
    if total is not None:
        pvals = p_values.pvalue.values.tolist() + [1]*(total-len(p_values))
        idx = p_values.pvalue.index.tolist() + [None]*(total-len(p_values))
    else:
        pvals = p_values.pvalue.values
        idx = p_values.pvalue.index

    keep,qvals = fdrcorrection(pvals,alpha=fdr_rate)
    
    result = p_values.copy()
    result['qvalue'] = qvals[:len(p_values)]
    result = result[keep[:len(p_values)]]
   
    return result.sort_values('qvalue')

def compute_threshold(S,k,cutoff):
    """Computes kurtosis-based threshold for a component of an S matrix
        S: Component matrix with gene weights
        k: Component name
        cutoff: Minimum test statistic value to determine threshold
    """
    i = 0
    # Sort genes based on absolute value
    ordered_genes = abs(S[k]).sort_values()
    K,p = stats.normaltest(S.loc[:,k])
    while K > cutoff:
        i -= 1
        # Check if K statistic is below cutoff
        K,p = stats.normaltest(S.loc[ordered_genes.index[:i],k])
    comp_genes = ordered_genes.iloc[i:]
    if len(comp_genes) == len(S.index):
        return max(comp_genes)+.05
    else:
        return np.mean([ordered_genes.iloc[i],ordered_genes.iloc[i-1]])
        
def get_regulon_enrichment(genes,all_genes,reg_name,trn):
    """ Calculates enrichment statistics for a specific regulon
        genes: List of genes
        reg_name: Regulator (single or multiple, where "/" = OR and "+" = AND)
        all_genes: List of all genes in organism
        trn: Dataframe containing transcriptional regulatory network
    """
    
    # Parse regulator
    
    if '+' in reg_name:
        join = set.intersection
        tfs = reg_name.split('+')
    elif '/' in reg_name:
        join = set.union
        tfs = reg_name.split('/')
    else:
        join = set.union
        tfs = [reg_name]
        
    # Combine regulon
    reg_genes = join(*[set(trn[trn.TF == tf].gene_id) for tf in tfs])
    
    # Calculate enrichment
    return single_enrichment_helper(reg_genes,set(genes),set(all_genes),reg_name)
    
        
def single_enrichment_helper(reg_genes,ic_genes,all_genes,reg_name):
    """ Calculates the enrichment of a set of genes in an i-modulon
        reg_genes: Genes in regulon
        ic_genes: Genes in an independent component
        all_genes: All genes in genome
        reg_name: Regulator name
    """
    # Compute enrichments
    ((tp,fp),(fn,tn)) = contingency(reg_genes,ic_genes,all_genes)
        
    # Handle edge cases
    if tp == 0:
        res = [0,1,0,0,0]
    elif fp == 0 and fn == 0:
        res = [np.inf,0,1,1,len(ic_genes)]
    else:
        odds,pval = stats.fisher_exact([[tp,fp],[fn,tn]],alternative='greater')
        recall = np.true_divide(tp,tp+fn)
        precision = np.true_divide(tp,tp+fp)
        res = [np.log(odds),pval,recall,precision,tp]
    return pd.Series(res,index=['log_odds','pvalue','recall','precision','TP'],
                     name=reg_name)
 

def multi_enrichment_helper(ic_genes,all_genes,trn,n_tfs=1,type='and'):
    """ Helper function to identify enrichment of multiple regulators
        ic_genes: Genes in independent component
        all_genes: All genes in genome
        n_tfs: Maximum number of simultaneous TF enrichments
        trn: TRN
        type: "and" to check for enrichment in all regulators, "or" to
            check for enrichment in any regulator
    """
    
    if type == 'and':
        join = set.intersection
        strjoin = '+'.join
    elif type == 'or':
        join = set.union
        strjoin = '/'.join
    else:
        raise ValueError("Type must be 'and' or 'or'")

    # Only search regulators that regulate at least one gene in component
    ic_regs = set(trn[trn.gene_id.isin(ic_genes)].TF)
    
    # If no TFs, return empty DF
    if len(ic_regs) < n_tfs:
        return pd.DataFrame(columns=['log_odds','pvalue','recall','precision'])

    enrichment_list = []
    
    # Compute enrichments for all combinations
    for tfs in combinations(ic_regs,n_tfs):   
        reg_genes = join(*[set(trn[trn.TF == tf].gene_id) for tf in tfs])
        reg_name = strjoin(tfs)
    
        enrichment_list.append(single_enrichment_helper(reg_genes,ic_genes,
                                               all_genes,reg_name))
    DF_enriched = pd.concat(enrichment_list,axis=1).T
    return DF_enriched


def compute_enrichments(genes,all_genes,trn,max_tfs,fdr_rate=0.01):
    """Calculate regulon enrichments for a set of genes
        genes: Set of genes to check for regulon enrichments
        all_genes: All genes in genome
        trn: Dataframe of gene_id and TF
        max_tfs: Maximum number of TFs in a single enrichment
        fdr_rate: False detection rate
    """
    # Initialize variables
    list2struct = []
    m = 0
    
    # Calculate enrichment of single TF
    list2struct.append(multi_enrichment_helper(set(genes),set(all_genes),trn,1))
    m += len(trn.TF.unique())
    
    # Calculate combined enrichments
    for n in range(2,max_tfs+1):
        list2struct.append(multi_enrichment_helper(set(genes),set(all_genes),trn,n,'and'))
        list2struct.append(multi_enrichment_helper(set(genes),set(all_genes),trn,n,'or'))
        # Total number of comparisons includes combinations of all TFs twice
        # (once for "and" and once for "or")
        m += special.comb(len(trn.TF.unique()),n)*2
    
    # To reduce computation time, truncate p-value list
    pvalues = pd.concat([df[df.pvalue < .5] for df in list2struct])

    # Run false detection, using total number of comparisons
    df = FDR(pvalues,fdr_rate,total=int(m))
    df['f1score'] = 2*(df['precision']*df['recall'])/(df['precision']+df['recall'])
    df['n_tf'] = [s.count('/') + s.count('+')+1 for s in df.index]
    return df.sort_values(['f1score','n_tf'],ascending=[False,True])
    
###############
## Load Data ##
###############


class load_data(object):
    """Contains all data and information relating to the independent gene 
    components and RNAseq data."""
    def __init__(self,X,S,A,metadata,
                 annotation=None,trn=None,fasta=None,cutoff=500,
                 names=None,organism=None,dima=True):

        """Required Args:
             X: RNAseq data in log(TPM+1) units
             S: S matrix from ICA.
             A: A matrix from ICA.
             metadata: Metadata table
           Optional Args:
             annotation: Genome information table with any of the columns:
                 [start,stop,strand,gene_name,length,product,operon,cog]
             trn: TRN table with regulator names and regulated gene IDs.
             cutoff: kurtosis cutoff
             names: Filename, dict, or series representing modulon names
             dima: If true, compute distributions for differential 
                 i-modulon activity (must be false with no reps) ## TODO: autodetect this
        """
        
        ## Load Data        
        self.S = pd.read_csv(S,index_col=0)
        self.A = pd.read_csv(A,index_col=0)
        try:
            self.S.columns = self.S.columns.astype(int)
            self.A.index = self.A.index.astype(int)
        except:
            pass
        
        self.X = pd.read_csv(X,index_col=0)
        
        
        ## Load Metadata
        self.metadata = pd.read_csv(metadata,index_col=0)
        
        # Check consistency between S,A,X and metadata
        def unique(A,B):
            return sorted(set(A).union(set(B)) - set(A).intersection(set(B)))
            
        if set(self.S.index) != set(self.X.index):
            missing_genes = ', '.join(unique(self.S.index,self.X.index))
            warnings.warn('Inconsistent genes in S and X: {}'.format(missing_genes))
            
        if set(self.A.columns) != set(self.X.columns):
            missing_reps = ', '.join(unique(self.A.columns,self.X.columns))
            warnings.warn('Inconsistent samples in X and A: {}'.format(missing_reps))
        
        if set(self.X.columns) != set(self.metadata.index):
            missing_reps = ', '.join(unique(self.X.columns,self.metadata.index))
            warnings.warn('Inconsistent metadata for: {}'.format(missing_reps))
            
        # Ensure proper data order
        self.X = self.X[self.metadata.index]
        self.S = self.S.reindex(self.X.index)
        self.A = self.A[self.metadata.index]
        
        ## Load i-modulon names
        if names is None:
            _names = self.S.columns
        elif isinstance(names,list):
            if len(names) != len(self.S.columns):
                raise ValueError('Invalid number of names')
            else:
                _names = names
        elif isinstance(names,pd.Series):
            if len(names) != len(self.S.columns):
                raise ValueError('Invalid number of names')
            else:
                _names = names.tolist()
        else:
            _names = self.S.columns
        
        # Ensure that each name is unique by append _1 to duplicates
        def check_names(namelist):
            counts = dict()
            final_list = []
            for name in namelist:
                if name in counts.keys():
                    final_list.append('{}_{}'.format(name,counts[name]))
                    counts[name] += 1
                else:
                    final_list.append(name)
                    counts[name] = 1
            return final_list

        self.names = check_names(_names)
        
        self.S.columns = self.names
        self.A.index = self.names
        
        ## TODO: Add stats file
        
        ## Fit distributions to components for DMA
        if dima:
            try:
                _diff = pd.DataFrame()
                for name,group in self.metadata.groupby(['project_id','condition_id']):
                    for i1,i2 in combinations(group.index,2):
                        _diff['__'.join(name)] = abs(self.A[i1]-self.A[i2])

                self.dist = {}
                for k in self.names:
                    self.dist[k] = stats.lognorm(*stats.lognorm.fit(_diff.loc[k].values)).cdf
            except:
                raise Exception('No replicates detected. Try running with dima=False')



        ## Load gene annotation information
        if annotation is None:
            self.gene_info = None
            self.cog_colors = None
            self.gene_cogs = None
            self.gene_colors = None
        else:
            self.gene_info = pd.read_csv(annotation,index_col=0,
                               dtype = {'start':int,'stop':int,'length':int})

            self.num2name = self.gene_info.gene_name.to_dict()
            self.name2num = {v:k for k,v in self.num2name.items()}
            
            if organism == 'ecoli':
                self.name2num.update({'cecR':'b0796','comR':'b1111','dan':'b3060',
                                      'appY':'b0564','dinJ;yafQ':'b0226',
                                      'dpiA':'b0620','envR':'b3264','ihf':'b1712',
                                      'flhD;flhC':'b1892',
                                      'gadE;rcsB':self.name2num['gadE'],
                                      'btsR':'b2125','gutM':'b2706','gutR':'b2707',
                                      'h-NS':'b1237','hU':'b4000',
                                      'higB;higA':'b3083',
                                      'hipA;hipB':self.name2num['hipA'],
                                      'matA':'b0294','nimR':'b1790',
                                      'ntrC':'b3868','ntrc':'b3868',
                                      'pdeL':'b0315',
                                      'rcsA;rcsB':self.name2num['rcsA'],
                                      'rcsB;bglJ':'b4366',
                                      'sutR':'b1434','rpoF':'b1922',
                                      'Sigma19':'b4293','Sigma24':'b2573',
                                      'Sigma28':'b1922','Sigma32':'b3461',
                                      'Sigma38':'b2741','Sigma54':'b3202',
                                      'gatR':'b2087_1','Sigma70':'b3067',
                                      'fis':'b3261','ihfA;ihfB':'b1712',
                                      'glpR':'b3423','relB;relE':'b1564',
                                      'TPP':None,'histidine':None,
                                      'adenosylcobalamin':None,'manganese':None,
                                      'FMN':None,'gcvB':'b4443'})
            
            # Ensure that the gene_info table only has genes in X
            self.gene_info = self.gene_info.reindex(self.X.index)
            # Throw a warning if any genes are in X and not gene_info
            if sum(self.gene_info.gene_name.isnull()) > 0:
                missing_genes = ', '.join(self.gene_info[self.gene_info.gene_name.isnull()].index)
                warnings.warn('Genes missing annotations: {}'.format(missing_genes))

            self.cog_colors = dict(zip(self.gene_info['cog'].unique().tolist(), 
                       ['red','pink','y','orchid','mediumvioletred','green',
                        'lightgray','lightgreen','slategray','blue',
                        'saddlebrown','turquoise','lightskyblue','c','skyblue',
                        'lightblue','fuchsia','dodgerblue','lime','sandybrown',
                        'black','goldenrod','chocolate','orange']))

            self.gene_cogs = self.gene_info.cog.to_dict()
            self.gene_colors = {k:self.cog_colors[v] for k,v \
                                in self.gene_cogs.items()}
        
                
        ## Load TRN
        if trn is None:
            self.trn = None
            self.trn_dict = None
        else:
            self.trn = pd.read_csv(trn,index_col=0)
            self.trn = self.trn[self.trn.gene_id.isin(self.X.index)]
            self.trn_dict = {name:','.join(group.TF.values) 
                        for name,group in self.trn.groupby('gene_id')}
            

        
        ## Load miscellaneous
        if (isinstance(self.gene_info,pd.DataFrame)) and \
           ('operon' in self.gene_info.columns):
            self.all_operons = set(self.gene_info.operon)
        else:
            self.all_operons = None
            
        self.regulon_mapping = {}
        self.motif_info = {}
        
        
        # Load fasta file
        if fasta is None:
            self.fasta = None
        else:
            with open(fasta, "rU") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    self.fasta = record
    
    
        ## Define cutoffs
        self.cutoff = cutoff
        self.thresholds = {k:compute_threshold(self.S,k,self.cutoff) \
                            for k in self.S.columns}

    # Get component information
    def component_DF(self,k,tfs=[]):
        df = pd.DataFrame(self.S[k].sort_values())
        df.columns = ['comp']
        if self.gene_info is not None:
            if 'product' in self.gene_info.columns:
                df['product'] = self.gene_info['product']
            if 'gene_name' in self.gene_info.columns:
                df['gene_name'] = self.gene_info['gene_name']
            if 'operon' in self.gene_info.columns:
                df['operon'] = self.gene_info['operon']
            if 'length' in self.gene_info.columns:
                df['length'] = self.gene_info.length
                
        if self.trn_dict is not None:
            df['TF'] = [self.trn_dict[gene] 
                        if gene in self.trn_dict.keys() 
                        else '' for gene in df.index]
            for tf in tfs:
                df[tf] = [tf in regs.split(',') for regs in df['TF']]
                
        return df.sort_values('comp')
        
    def show_enriched(self,k):
        df_gene = self.component_DF(k)
        return df_gene[abs(df_gene.comp) > self.thresholds[k]]

    def genes2operons(self,genes):
        if self.gene_info is None:
            raise ValueError("Annotation information not provided")
        elif 'operon' not in self.gene_info.columns:
            raise ValueError("Operon data not in genome annotation")
        else:
            return set([self.gene_info.loc[gene,'operon'] for gene in genes])
