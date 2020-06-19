import os,re,six
# import escher,cobra
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2
from adjustText import adjust_text

from icaviz.load import *
from icaviz.util import *

sns.set_style('whitegrid')



##########################
## Component Gene Plots ##
##########################

def plot_genes(ica_data,k,labels='auto',adjust=True,figsize=(7,5),ax=None):

    """Display the mean expression vs. component value for each gene.
    
    
    Args:
        ica_data: ICA data object.
        k: name of the component.
        labels: True to show gene labels, False to hide.
        adjust: Use adjust_text to improve readability (default: True)
        figsize: Figure size if ax is None (default: (15,7))
        ax: Predefined axis to plot on (default: None).
    """
    
    if ax == None:
        fig,ax = plt.subplots(figsize=figsize)
    
    # Define colors from COG (if available)
    if ica_data.gene_colors is None:
        colors = 'b'
    else:
        colors = [ica_data.gene_colors[gene] for gene in ica_data.S.index]
    
    # Draw scatterplot
    scatter = ax.scatter(ica_data.X.mean(axis=1).values,
                         ica_data.S[k].values,
                         c=colors,s=20,
                         alpha=0.7,linewidth=0.0)
    
    ax.set_xlabel('Mean Expression',fontsize=20)
    ax.set_ylabel('I-Modulon Gene Weights',fontsize=20)
 
    # Get axes bounds
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    
    # Draw horizontal dashed lines
    if ica_data.thresholds[k] != 0:
        ax.hlines([ica_data.thresholds[k],-ica_data.thresholds[k]],xmin,xmax,colors='k',linestyles='dashed',
                  linewidth=1)
                  
    # Make sure axes bounds stay constant
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
              
    # Add labels on datapoints        
    component_genes = ica_data.S[abs(ica_data.S[k]) >= ica_data.thresholds[k]][k].index
    texts = []
    expand_args = {'expand_objects':(1.2,1.4),
                   'expand_points':(1.3,1.3)}
    
    ## Put gene name if component contains under 20 genes
    if labels == True or (labels != False and len(component_genes) < 20):
        for gene in component_genes:
            texts.append(ax.text(ica_data.X.loc[gene].mean(),
                                 ica_data.S.loc[gene,k],
                                 ica_data.num2name[gene],
                                 fontsize=12))
        expand_args['expand_text'] =(1.4,1.4)
                                 
    ## Repel texts from other text and points
    rect = ax.add_patch(Rectangle((xmin,-ica_data.thresholds[k]),xmax-xmin,2*ica_data.thresholds[k],fill=False,
                     linewidth=0))
    if adjust:
        adjust_text(texts,ax=ax,add_objects = [rect],
                    arrowprops=dict(arrowstyle="-",color='k',lw=0.5),
                    only_move={'objects':'y'},**expand_args)
                             

    # Add legend
    legend_info = []
    if ica_data.gene_info is None or 'cog' not in ica_data.gene_info.columns:
        ax.plot([],[],"o",color='b',markersize=5,
                label='{} genes in I-modulon'.format(len(ica_data.show_enriched(k))))
        leg = ax.legend(loc='upper left',bbox_to_anchor=(0,-0.15),fontsize=12)
    else:
        for name,group in ica_data.gene_info.groupby('cog'):
            # Get number of genes in COG
            cog_genes = group.index
            num_genes = sum(abs(ica_data.S.loc[cog_genes,k]) >= ica_data.thresholds[k])

            if num_genes > 0:
                text = '{} ({})'.format(name,num_genes)
                
                # Create legend patch and add to list
                legend_info.append((ica_data.cog_colors[name],text,num_genes))
            
        ## Sort legend by number of genes
        legend_info = sorted(legend_info,key=lambda x: x[2],reverse=True)
        
        ## If no legend entries, return
        if len(legend_info) == 0:
            return ax
        ## If over 6 legend entries, only print first 9 plus "Other"
        elif len(legend_info) > 6:
            num_genes = sum([row[2] for row in legend_info[5:]])
            legend_info = legend_info[:5] + [('white','Other ({})'.format(num_genes,num_genes))]
        
        ## Add legend entries to plot as empty    
        for info in legend_info:
            ax.plot([],[],"o",color=info[0],markersize=5,label=info[1])

        
        leg = ax.legend(loc='upper left',bbox_to_anchor=(0,-0.15),
                        fontsize=12,title='COG groupings')

    leg._legend_box.align = "left"
    leg.get_title().set_fontweight('bold')
    leg.get_title().set_fontsize(14)
    return ax

def plot_samples_bar(ica_data,k,project=None,ax=None,legend_args={}):

    # Check that i-modulon exists
    if k not in ica_data.names:
        raise ValueError('Component does not exist: {}'.format(k))
    
    if ax == None:
        fig,ax = plt.subplots(figsize=(15,2))
    
    
    # Get ymin and max
    ymax = ica_data.A.loc[k].max()+3
    ymin = ica_data.A.loc[k].min()-3
    
    # Plot all projects not in the highlighted set
    other = ica_data.metadata[ica_data.metadata.project_id!=project].reset_index()
    ax.bar(range(len(other)),ica_data.A.loc[k,other['sample_id']],
            width=1,linewidth=0,align='edge',label='Previous Experiments')
    
    # Draw lines to discriminate between projects
    p_lines = other.project_id.drop_duplicates().index.tolist() + [len(other),len(ica_data.metadata)]
    ax.vlines(p_lines,ymin,ymax,colors='lightgray',linewidth=1)
    
    # Add project labels
    move = True
    locs = (np.array(p_lines)[:-1] + np.array(p_lines)[1:])/2
    
    for loc,name in zip(locs,other.project_id.drop_duplicates().tolist()+[project]):
        ax.text(loc,ymax+2+move*4,name,fontsize=12,
                horizontalalignment='center')
        move = not move
    
    
    # Plot project of interest
    idx = len(other)
    for name,group in ica_data.metadata[ica_data.metadata.project_id==project].groupby('condition_id'):
        values = ica_data.A.loc[k,group.index].values
        ax.bar(range(idx,idx+len(group)),values,width=1,
                linewidth=0,align='edge',label=name)
        idx +=len(group)
        
    # Make legend
    args = {'loc':2,'ncol':7,'bbox_to_anchor':(0,0)}
    args.update(legend_args)
    ax.legend(**args)
    
    # Prettify
    ax.set_xticklabels([])
    ax.grid(False,axis='x')
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([0,ica_data.A.shape[1]])
    return ax

### OLD VERSION MAY BE USEFUL LATER???
'''
def plot_samples_bar(ica_data,k):

    # Define sort function for conditions
    def keyfxn(x):
        name = x[0]
        match = re.match('^.+?ale(\d+)$',name)
        if match:
            return '%02d'%int(match.group(1))
        elif name.startswith('wt') or name.startswith('glu4'):
            return '00'+name[3:]
        else:
            return name

    # Set order for display
    proj_order = ['base','fur','gadewx','oxidative','nac_ntrc','ompr','misc',
                  'me_param','minspan','cra_crp','rpoB','crp','glu','42c','ssw','pgi','ica']
                  
    leftover = set(ica_data.metadata.project_id.unique()) - set(proj_order)
    proj_order += list(leftover)
    
    list2struct = []
    for proj in proj_order:
        group1 = ica_data.metadata[ica_data.metadata.project_id == proj]
        for cond,group2 in sorted(group1.groupby('condition_id'),key=keyfxn):
            for name in group2.sample_id:
                list2struct.append([proj,cond,ica_data.A.loc[k,name],len(group2)])
    DF_comp = pd.DataFrame(list2struct,columns = ['project_id','condition_id','value','length'])
    DF_comp['width'] = 1/DF_comp.length
    DF_comp.index = [0]+np.cumsum(DF_comp.width).tolist()[:-1]

    # Get xlabels and tick marks
    xticks = []
    xticklabels = []
    vlines = []
    proj_labels = []
    proj_locs = []
    for proj,group1 in DF_comp.groupby('project_id'):
        vlines.append(min(group1.index))
        proj_labels.append(proj)
        proj_locs.append(min(group1.index)+group1.width.sum()/2)
        for cond,group2 in group1.groupby('condition_id'):
            xticks.append(np.mean(group2.index)+np.mean(group2.width)/2)
            xticklabels.append(cond)

    fig,ax = plt.subplots(figsize=(20,5))
    ax.bar(DF_comp.index,DF_comp.value,width = DF_comp.width,align='edge')
    ax.set_xlim([0,len(DF_comp)])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=10)
    ax.tick_params(axis='x',which='major',direction='in',length=5,labelrotation=90)
    ax.grid(False,which='major')
    ylim = ax.get_ylim()
    ax.vlines(vlines,-100,100,colors='lightgray',linewidth=1)
    ax.set_ylim(ylim)
    ax.set_xlim(DF_comp.index.min(),DF_comp.index.max()+DF_comp.width.iloc[-1])
    ax.set_title(k,fontsize=20,y=1.08)

    # # Add project labels
    ax2 = ax.twiny()
    ax2.grid(False)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(proj_locs)
    ax2.set_xticklabels(proj_labels,fontsize=14)
    ax2.xaxis.tick_top()
    return ax
'''


#############################
## Regulon-component Plots ##
#############################

def get_regulon_mapping(ica_data,k,tf=None,fdr_rate=1e-5): #TODO expand to multi-TF
    if tf is None:
        if k not in ica_data.regulon_mapping.keys():
            ic_genes = set(ica_data.show_enriched(k).index)
            all_genes = set(ica_data.X.index)
            ica_data.regulon_mapping[k] = compute_enrichments(ic_genes,
                                    all_genes,ica_data.trn,1,fdr_rate)
        return ica_data.regulon_mapping[k]
    
    else:
        reg_genes = set(ica_data.trn[ica_data.trn.TF==tf].gene_id)
        ic_genes = set(ica_data.show_enriched(k).index)
        all_genes = set(ica_data.X.index)
        return pd.DataFrame(single_enrichment_helper(reg_genes,ic_genes,all_genes,tf)).T


def plot_regulon(ica_data,k,tf=None,num_regs=1):
    
    tf_pvals = get_regulon_mapping(ica_data,k,tf)
    tf_pvals = tf_pvals.iloc[:min(len(tf_pvals),num_regs)]
    
    axes = []
    for tf,row in tf_pvals.iterrows():
        ax1 = regulon_histogram(ica_data,k,tf)
        ax2 = regulon_venn(ica_data,k,tf)
        ax3 = regulon_scatter(ica_data,k,tf)
        axes.append((ax1,ax2,ax3))
    return axes

def regulon_histogram(ica_data,k,tf=None,bins=20,ax=None):
    """
    Plot a histogram of I-modulon gene weights, colored by regulator binding.
    Args:
        ica_data: ICA data object.
        k: Name of the component.
        tf: Regulon to color (default: regulon with lowest p-value)
        ax: Predefined axis to plot on (default: None).
    
    
    """
    if ica_data.regulon_mapping is None:
        raise ValueError('No TRN information found')
        
    DF_gene = ica_data.component_DF(k,tfs=[tf])

    tf_info = get_regulon_mapping(ica_data,k,tf)
    if len(tf_info) == 0:
        tf_info = pd.Series({'pvalue':1,'precision':0,'recall':0})
    else:
        tf_info = tf_info.iloc[0]
    # Compute optimal range for histogram
    xmin = min(min(DF_gene.comp),-ica_data.thresholds[k])
    xmax = max(max(DF_gene.comp),ica_data.thresholds[k])
    width = 2*ica_data.thresholds[k]/(np.floor(2*ica_data.thresholds[k]*bins/(xmax-xmin)-1))
    xmin = -ica_data.thresholds[k]-width*np.ceil((-ica_data.thresholds[k] - xmin)/width)
    xmax = xmin + width*bins
    
    # Initialize figure
    if ax == None:
        fig,ax = plt.subplots(figsize=(14,5))
        
    ax.set_yscale('log', nonposy='clip')
    ax.xaxis.grid(False)
    ax.set_xlabel('I-Modulon Gene Weights',fontsize=16)
    ax.set_ylabel('Count (log scale)',fontsize=16)
    
    # Plot histogram for genes unregulated by TF
    ax.hist(DF_gene[~DF_gene[tf]].comp,color='#aaaaaa',alpha=0.7,bins=bins,
            range=(xmin,xmax),linewidth=0,
            label='Unregulated\nby ' + tf)
    
    # Plot histogram for genes regulated by TF
    ax.hist(DF_gene[DF_gene[tf]].comp,color='salmon',alpha=0.7,bins=bins,
            range=(xmin,xmax),linewidth=0,
            label='Regulated\nby ' + tf)
    
    # Add vertical lines       
    ymin,ymax=ax.get_ylim()
    if six.PY2:
        labelstr = 'P-value = {:.2e}\nPrecision = {:.0f}%%\nRecall = {:.0f}%%'.format(tf_info.pvalue,\
                                               tf_info.precision*100,\
                                               tf_info.recall*100)
    else:
        labelstr = 'P-value = {:.2e}\nPrecision = {:.0f}%\nRecall = {:.0f}%'.format(tf_info.pvalue,\
                                               tf_info.precision*100,\
                                               tf_info.recall*100)
    ax.vlines([ica_data.thresholds[k],-ica_data.thresholds[k]],0,100,linestyles='dashed',
               label=labelstr)
                          
    # Add gene names to each bin
    eps = 1e-5
    for x in np.arange(xmin,xmax,width):
        # Only add names to bins outside the cutoff
        if x <= -ica_data.thresholds[k]-eps or x >= ica_data.thresholds[k]-eps:
            # Get genes in each bin
            bin_genes = DF_gene[(x < DF_gene.comp) & 
                                (DF_gene.comp < x+width)]

            # Define starting y value for text
            y = max(sum(bin_genes[tf]),sum(~bin_genes[tf]))
            y *= 1.1
            
            # If more than 13 genes, keep only largest 12 and include "+X"
            if len(bin_genes) > 13:
                leftover = bin_genes.reindex(abs(bin_genes.comp).sort_values()[:-12].index)
                bin_genes = bin_genes.reindex(abs(bin_genes.comp).sort_values()[-12:].index)
                bin_genes.loc['Other',tf] = None
                bin_genes.loc['Other','gene_name'] = '+{}'.format(len(leftover))
            
            # Draw text
            for i,row in bin_genes.iterrows():
                if pd.isnull(row[tf]):
                    c = 'black'
                elif row[tf]:
                    c = 'darkred'
                else:
                    c = '#666666'
                ax.text(x+width/2,y,row['gene_name'],color=c,
                        ha='center',va='bottom',fontsize=10,fontweight='bold')
                y *= 1.45
    
    # Draw legend
    legend = ax.legend(fontsize=12,frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_alpha(1)
    
    return ax
        
        
def regulon_venn(ica_data,k,tf,ax=None):
    if ica_data.regulon_mapping is None:
        raise ValueError('No TRN information found')
    
    if ax == None:
        fig,ax = plt.subplots(figsize=(5,5))
    
    # Take care of and/or enrichments
    if '+' in tf:
        reg_list = []
        for tfx in tf.split('+'):
            reg_list.append(set(ica_data.trn[ica_data.trn.TF==tfx].gene_id.unique()))
        reg_genes = set.intersection(*reg_list)
    elif '/' in tf:
        reg_genes = set(ica_data.trn[ica_data.trn.TF.isin(tf.split('/'))].gene_id.unique())
    else:
        reg_genes = set(ica_data.trn[ica_data.trn.TF==tf].gene_id.unique())
    
    # Get component genes and operons
    comp_genes = set(ica_data.show_enriched(k).index)
    
    reg_operons = len(ica_data.genes2operons(reg_genes-comp_genes))
    comp_operons = len(ica_data.genes2operons(comp_genes-reg_genes))
    both_operons = len(ica_data.genes2operons(reg_genes & comp_genes))
    
    # Draw venn diagram and resize texts
    venn = venn2((reg_genes,comp_genes),
                 set_labels=('Regulon\nGenes','I-Modulon\nGenes'),
                 ax=ax)
    for text in venn.set_labels:
        text.set_fontsize(20)
        if text.get_text() == u'Regulon\nGenes':
            text.set_color('darkred')
        else:
            text.set_color('darkgreen')
    [reg_venn,comp_venn,both_venn] = venn.subset_labels

    # Add operon numbers to labels
    comp_venn.set_fontsize(20)
    comp_venn.set_text(comp_venn.get_text()+'\n({})'.format(comp_operons))
    reg_venn.set_fontsize(20)
    reg_venn.set_text(reg_venn.get_text()+'\n({})'.format(reg_operons))
    if both_venn is not None:
        both_venn.set_fontsize(20)
        both_venn.set_text(both_venn.get_text()+'\n({})'.format(both_operons))

    return ax

def regulon_scatter(ica_data,k,tf,ax=None):
    if ica_data.regulon_mapping is None:
        raise ValueError('No TRN information found')
    
    # Initialize axis
    if ax==None:
        fig,ax = plt.subplots(figsize=(7,5))
    
    # Empty axis if no expression data for TF
    if (tf not in ica_data.name2num.keys()) or \
       (ica_data.name2num[tf] is None) or \
       (ica_data.name2num[tf] not in ica_data.X.index):
        ax.text(0.5,0.5,'Data unavailable for {}'.format(tf),fontsize=20,ha='center')
        ax.grid(False)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        return ax

    
    # Get metadata clusters
    # clusters = cluster_metadata(ica_data,k)
    
#     # Find experiments that are outliers in the mode
#     names = []
#     outlier_cols = []
#     ub = clusters[k].mean() + 3*clusters[k].std()
#     lb = clusters[k].mean() - 3*clusters[k].std()
#     for name,group in clusters.groupby('group'):
#         mean = group[k].mean()
#         if mean > ub or mean < lb:
#             names.append(name)
#             outlier_cols += group['exp_id'].tolist()

    # Separate KO experiments
    ko_cols = [exp for exp in ica_data.A.columns if 'del'+tf.lower() in exp \
               or 'del_'+tf.lower() in exp]
    other_cols = set(ica_data.A.columns) - set(ko_cols) #- set(outlier_cols) 
    
    # Make scatter plot only if TF expression is in our dataset
    ax.scatter(ica_data.X.loc[ica_data.name2num[tf],other_cols],
               ica_data.A.loc[k,other_cols],
               label='Conditions')
    
#         # Add colors to experimental outliers
#         if len(outlier_cols) > 0:
#             ax.scatter(ica_data.A.loc[k,outlier_cols],
#                        ica_data.X.loc[ica_data.name2num[tf],outlier_cols],
#                        label='Exp Outliers')

    # Add colors to TF KO experiments
    if len(ko_cols) > 0:
        ax.scatter(ica_data.X.loc[ica_data.name2num[tf],ko_cols],
                   ica_data.A.loc[k,ko_cols],
                   label='{} KO'.format(tf))
    
    # Draw best fit line
    fit_line(ica_data.X.loc[ica_data.name2num[tf],other_cols],
             ica_data.A.loc[k,other_cols],
             ax)

    ax.set_ylabel('I-Modulon Activity',fontsize=16)
    ax.set_xlabel('{} Expression Level'.format(tf),fontsize=16)
    
    # Draw and format legend
    legend = ax.legend(frameon=True,fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1)
        
    return ax

def fit_line(x,y,ax,xlim=None,label=True):
    if xlim==None:
        xlim=[min(x),max(x)]
    label = 'Pearson R = {:.2f}\np-value = {:.2e}'.format(*stats.pearsonr(x,y))
    if label:
        ax.plot(xlim, np.poly1d(np.polyfit(x,y,1))(xlim),c='k',
                linestyle='--',label=label)
    else:
        ax.plot(xlim, np.poly1d(np.polyfit(x,y,1))(xlim),c='k',linestyle='--')
        
#################
## Escher Plot ##
#################

def escher_plot(ica_data,k,display='notebook',
                map_file='models/iML1515_map.json',
                model_file='models/iML1515.json',
                detailed=True):
                
    """Display components on an Escher map. Significant genes with positive
       and negative weights will be colored red and blue, respectively.
    
    
    Args:
        ica_data: ICA data object.
        k: name of the component.
        display: 'notebook' or 'browser'
        map_file: JSON file with map data
        model_file: JSON file with model data
        detailed: Show secondary metabolites (default: True)
        ax: Predefined axis to plot on (default: None).
    """
    
    
    if not os.path.isfile(map_file):
        raise ValueError('Map file not found:',map_file)
    if not os.path.isfile(model_file):
        raise ValueError('Model file not found:',model_file)            
    
    
    # Get i-modulon weights
    comp_weights = ica_data.S[k]
    cutoff = ica_data.thresholds[k]
    
    print('I-Modulon cutoff:',cutoff)
        
    model = cobra.io.load_json_model(model_file)
    default_args = {'hide_secondary_metabolites':not detailed,
                    'reaction_scale':[{'type':'min','color':'#0000ff','size':25},
                                      {'type':'value','value':-cutoff-.001,'color':'#0000ff','size':25},
                                      {'type':'value','value':-cutoff+.001,'color':'#c8c8c8','size':5},
                                      {'type':'value','value':0,'color':'#c8c8c8','size':5},
                                      {'type':'value','value':cutoff-.001,'color':'#c8c8c8','size':5},  
                                      {'type':'value','value':cutoff+.001,'color':'#ff0000','size':25},
                                      {'type':'max','color':'#ff0000','size':25}],
                    'reaction_styles':['color','size'],
                    'hide_all_labels':False,
                    'show_gene_reaction_rules':True
        }
    
    # Convert gene weights to reaction weights, using max value in GPR
    rxn_weights = {}
    for rxn in model.reactions:
        genes = [g for gene in rxn.genes for g in (gene.id,gene.id+'_1',gene.id+'_2',gene.id+'_3')]
        if len(genes) == 0:
            rxn_weights[rxn.id] = 0
        else:
            max_gene = abs(comp_weights.reindex(genes).fillna(0)).idxmax()
            if max_gene in comp_weights.index:
                rxn_weights[rxn.id] = comp_weights.loc[max_gene]
            else:
                rxn_weights[rxn.id] = 0
    
    # Draw map
    escher_map = escher.Builder(map_json=map_file,**default_args)
    escher_map.reaction_data = rxn_weights
    if display == 'notebook':
        return escher_map.display_in_notebook()
    elif display == 'browser':
        return escher_map.display_in_browser()
    else:
        return escher_map
        
        
######################
## Compare ICA runs ##
######################

from graphviz import Digraph
from scipy.cluster.hierarchy import linkage,dendrogram
from tqdm import tqdm_notebook as tqdm

def compare_ica(S1,S2,metric='pearson',cutoff=0.2):
    
    # Only keep genes found in both S matrices
    common = set(S1.index) & set(S2.index)
    s1 = S1.reindex(common)
    s2 = S2.reindex(common)
    
    # Ensure column names are strings
    s1.columns = s1.columns.astype(str)
    s2.columns = s2.columns.astype(str)
    
    # Split names in half if necessary for comp1
    cols = {}
    for x in s1.columns:
        if len(x) > 10:
            cols[x] = x[:len(x)//2]+'-\n'+x[len(x)//2:]
        else:
            cols[x] = x
    s1.columns = [cols[x] for x in s1.columns]
    
    # Calculate correlation matrix
    corr = np.zeros((len(s1.columns),len(s2.columns)))
    
    for i,k1 in tqdm(enumerate(s1.columns),total=len(s1.columns)):
        for j,k2 in enumerate(s2.columns):
            if metric == 'pearson':
                corr[i,j] = abs(stats.pearsonr(s1[k1],s2[k2])[0])

    DF_corr = pd.DataFrame(corr,index=s1.columns,columns=s2.columns)

    # Initialize Graph
    dot = Digraph(engine='dot',graph_attr={'ranksep':'0.3','nodesep':'0','packmode':'array_u','size':'7,7'},
                  node_attr={'fontsize':'14','shape':'none'},
                  edge_attr={'arrowsize':'0.5'},format='png')
    
    # Set up linkage and designate terminal nodes
    loc1,loc2 = np.where(DF_corr > cutoff)
    links = list(zip(s1.columns[loc1],s2.columns[loc2]))
    
    if len(links) == 0:
        warnings.warn('No components shared across runs')
        return None,None
    
    # Initialize Nodes
    for k in sorted(s2.columns):
        if k in s2.columns[loc2]:
            color= 'black'
            font = 'helvetica'
        else:
            color = 'red'
            font = 'helvetica-bold'
        dot.node('data2_'+str(k),label=k,_attributes={'fontcolor':color,'fontname':font})
    
    for k in s1.columns:
        if k in s1.columns[loc1]:
            color = 'black'
            font = 'helvetica'
        else:
            color = 'red'
            font = 'helvetica-bold'
        dot.node('data1_'+str(k),label=k,_attributes={'fontcolor':color,'fontname':font})


    # Add links between related components
    for k1,k2 in links:
        width = DF_corr.loc[k1,k2]*5
        dot.edge('data1_'+str(k1),'data2_'+str(k2),_attributes={'penwidth':'{:.2f}'.format(width)})

    # Reformat names back to normal
    name1,name2 = list(zip(*links))
    inv_cols = {v:k for k,v in cols.items()}
    name_links = list(zip([inv_cols[x] for x in name1],name2))
    return dot,name_links

# Remove circos plot for now
#def circos_plot(ica_data,k,tf):
#    # Get component and regulon genes
#    comp_genes = ica_data.show_enriched(k).index.tolist()
#    reg_genes = ica_data.trn[ica_data.trn.TF == tf].gene_id.unique()

#    # Get TF genomic locations
#    tf_start = ica_data.gene_info.loc[ica_data.name2num[tf],'start']
#    tf_stop = ica_data.gene_info.loc[ica_data.name2num[tf],'stop']

#    # Get gene genomic locations
#    links = []

#     for gene in reg_genes:
#         gene_start = ica_data.gene_info.loc[gene,'start']
#         gene_stop = ica_data.gene_info.loc[gene,'stop']
#         locs = [tf_start,tf_stop,gene_start,gene_stop]
#         links.append('ecoli {:d} {:d} ecoli {:d} {:d} color=black,thickness=10p'.format(*locs))

#    for gene in comp_genes:
#        gene_start = ica_data.gene_info.loc[gene,'start']
#        gene_stop = ica_data.gene_info.loc[gene,'stop']
#        locs = [tf_start,tf_stop,gene_start,gene_stop]
#        links.append('ecoli {:d} {:d} ecoli {:d} {:d} color=black,thickness=10p'.format(*locs))
#       
#    # Write circos link text
#    with open('lib/circos/links.txt','w') as f:
#        circos = f.write('\n'.join(links))

#    # Run circos
#    circos_str = '/home/anand/bin/circos-0.69-6/bin/circos -conf \
#lib/circos/circos.conf -file {} -silent'
#    circos_img = 'lib/circos/{}.png'.format(k)
#    os.system(circos_str.format(circos_img))

#    # Plot output
#    fig,ax = plt.subplots(figsize=(5,5))
#    ax.grid(False)
#    ax.set_yticklabels([])
#    ax.set_xticklabels([])
#    with open(circos_img,'r') as f:
#        im = plt.imread(f)
#    ax.imshow(im,interpolation='quadric')
#    ax.set_title('Circos Plot',fontsize=20)



###############################
## Condition Component Plots ##
###############################
def plot_samples_box(ica_data,k,groups=None,max_groups=10,
                 figsize=(15,7),ax=None):

    """Display the mean expression vs. component value for each gene. If no
       grouping is predefined, it will group using the metadata table.
    
    
    Args:
        ica_data: ICA data object.
        k: name of the component.
        groups: Series defining sample grouping (default: None).
        max_groups: Max number of groups for auto-clustering.
        figsize: Figure size if ax is None (default: (15,7)).
        ax: Predefined axis to plot on (default: None).
    """
    
    if ax == None:
        fig,ax = plt.subplots(figsize=figsize)

    # Get metadata clustering
    if groups is not None:
        DF_mode = pd.DataFrame(groups)
        DF_mode.columns = ['group']
        DF_mode[k] = ica_data.A.loc[k]
    else:
        DF_mode = cluster_metadata(ica_data,k,max_groups=max_groups)
        
    DF_mode = DF_mode.sort_values(k)

    # Draw boxplot and make it transparent
    sns.boxplot(y=DF_mode[k],x=DF_mode['group'],ax=ax,showfliers=False,
                width=0.5)
    plt.setp(ax.artists, alpha=.5)
    sns.swarmplot(y=DF_mode[k],x=DF_mode['group'],ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right',fontsize=14)

    # Reverse y-limits because seaborn is weird and movetext doesn't understand
    ymax,ymin = ax.get_ylim()
    ax.set_ylim((ymin,ymax))

    # Set labels
    ax.set_xlabel('')
    ax.set_ylabel('I-Modulon Activity Level',fontsize=20)


    # TODO: Fix annotate
    if annotate:
        rows = zip(ax.collections,DF_mode.group.unique())
        # Add labels to points
        texts = []
        for collection,group in rows:
            points = collection.get_offsets()
            labels = DF_mode[DF_mode.group == group].index
            labelled_points = zip(points,labels)
            
            # Label points in the box as a whole
            mid_label = []
            uq,lq = np.percentile(points[:,0], [75 ,25],axis=0)
            mid_y = np.median(points[:,1])
            mid_x = points[:,0].mean()+0.25
            
            # Add labels
            for (x,y),label in labelled_points:
                if y >= uq or y <= lq:
                    texts.append(ax.text(x,y,label,fontname='Consolas'))
                else:
                    mid_label.append('{:-5d}'.format(label))
                    
            # Format and add middle box label
            if len(mid_label) > 0:
                cols = 6
                n_blanks = cols - len(mid_label)%cols
                mid_label_box = np.reshape(sorted(mid_label,key=int)+\
                                ['']*n_blanks,(-1,cols))
                mid_label_box = '\n'.join([''.join(row) for row \
                                in mid_label_box])
                texts.append(ax.text(mid_x,mid_y,mid_label_box,
                                     fontname='Consolas'))

        # Repel texts from other text and points
        adjust_text(texts,ax=ax,add_objects=ax.artists,
                    only_move={'objects':'x'},expand_objects=(1.3,1.2),
                arrowprops=dict(arrowstyle="-",color='k',lw=0.5))
    # Flip axis back to pseudonormal
    ax.set_ylim((ymax,ymin))
    
    return ax

################
## DIMA PLOTS ##
################

def plot_dima(ica_data,sample1,sample2,label=True,
              adjust=True,lfc=5,fdr_rate=0.1,ax=None):

    a1 = ica_data.A[sample1].mean(axis=1)
    a2 = ica_data.A[sample2].mean(axis=1)
    if ax is None:
        fig,ax = plt.subplots()
    
    # Compute DIMA
    df_diff = diff_act(ica_data,sample1,sample2,lfc=lfc,fdr_rate=fdr_rate)
    diff_mods = df_diff.index
    other_mods = list(set(ica_data.names) - set(diff_mods))
    
    # Create scatter plot
    ax.scatter(a1[diff_mods],a2[diff_mods],color='k')
    ax.scatter(a1[other_mods],a2[other_mods],color='gray',s=4,alpha=0.5)
    
    # Add axis labels
    ax.set_ylabel(re.search('__(.*)__',sample2[0]).group(1))
    ax.set_xlabel(re.search('__(.*)__',sample1[0]).group(1))
    
    # Add diagonal lines
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    allmin = min(xmin,ymin)
    allmax = max(xmax,ymax)
    ax.plot([allmin,allmax],[allmin,allmax],c='k',linestyle='dashed',linewidth=0.5)
    ax.plot([allmin,allmax-lfc],[allmin+lfc,allmax],c='gray',linestyle='dashed',linewidth=0.5)
    ax.plot([allmin+lfc,allmax],[allmin,allmax-lfc],c='gray',linestyle='dashed',linewidth=0.5)
    ax.set_xlim([allmin,allmax])
    ax.set_ylim([allmin,allmax])
    
    # Add labels
    if label:
        df_diff = pd.concat([df_diff,a1,a2],join='inner',axis=1)
        texts = []
        for k in df_diff.index:
            texts.append(ax.text(df_diff.loc[k,0],df_diff.loc[k,1],k,fontsize=5))
        if adjust:
            expand_args = {'expand_objects':(1.2,1.4),
                   'expand_points':(1.3,1.3)}
            expand_args['expand_text'] =(1.4,1.4)
            
            
            adjust_text(texts,ax=ax,
                    arrowprops=dict(arrowstyle="-",color='k',lw=0.5),
                    only_move={'objects':'y'},**expand_args)
    return ax
