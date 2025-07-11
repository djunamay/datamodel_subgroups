import glob
import numpy as np
from .plotting import plot_cat_continuous
import matplotlib.pyplot as plt
import pandas as pd
import glob


def extract_group_data(path, group_number):

    dictionary = dict(zip(['all_features_input', 
          'filtered_features_input', 
          'datamodel_input'], ['unfiltered F', 'filtered F', 'datamodel']))
    
    files = find_files_with_suffix(path, f'group_{group_number}_counterfactual_results')
    out = []
    for file in files:
        temp = pd.read_csv(file)
        temp['nclusters'] = int(file.split('_')[-5])
        out.append(temp)
    all_data = pd.concat(out)
    all_data['input'] = all_data['input'].map(dictionary)

    return all_data

def find_files_with_suffix(directory, suffix):
    pattern = f"{directory}/*{suffix}.csv"
    return sorted(glob.glob(pattern))

def extract_auc_diffs(all_data):

    dictionary = dict(zip(['all_features_input', 
          'filtered_features_input', 
          'datamodel_input'], ['unfiltered F', 'filtered F', 'datamodel']))
    
    df_wide = all_data.pivot_table(
        index=['nclusters','model_seed','input'],
        columns='prob_type',
        values=['mean_auc','mean_margins']
    )

    df_wide['mean_auc_diff'] = df_wide['mean_auc']['evaluation_on_split']-df_wide['mean_auc']['evaluation_outside_split']
    df_wide['mean_margins_diff'] = df_wide['mean_margins']['evaluation_on_split']-df_wide['mean_margins']['evaluation_outside_split']

    df_wide = df_wide.reset_index()
    
    return df_wide

def plot_auc_diffs(all_data, df_wide, tissue):

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), sharey=False, sharex=False)

    for i, name in enumerate(['evaluation_on_split','evaluation_outside_split']):
        data_subset = all_data[all_data['prob_type']==name]
        plot_cat_continuous(
                data=data_subset,
                cat_col='input',
                cont_col='mean_auc',
                test=None,
                text_format='full',
                hue = 'nclusters',
                #palette=['grey', 'blue', 'blue'],
                ax=axes[i],
                #y_lower=0.8,
                #y_upper=1.2,
                loc='outside',
                boxplot_kwargs={
                    'boxprops':     {'alpha': 0.5},
                    'whiskerprops': {'alpha': 0.5},
                    'capprops':     {'alpha': 0.5},
                    'medianprops':  {'alpha': 0.8},
                    'width': 0.8
                },
                order = ['unfiltered F', 'filtered F', 'datamodel'],
                stripplot_kwargs={'alpha': 0.6, 'size': 0}
            )
        
        val = data_subset[data_subset['input']=='datamodel']['mean_auc'].mean()
        axes[i].axhline(val, linestyle=':', label=f'$\mu_{'datamodel'}$', color='red')
        axes[i].get_legend().remove()
    

    plot_cat_continuous(
            data=df_wide,
            cat_col='input',
            cont_col='mean_auc_diff',
            test=None,
            text_format='full',
            hue = 'nclusters',
            #palette=['grey', 'blue', 'blue'],
            ax=axes[2],
            loc='outside',
            boxplot_kwargs={
                'boxprops':     {'alpha': 0.5},
                'whiskerprops': {'alpha': 0.5},
                'capprops':     {'alpha': 0.5},
                'medianprops':  {'alpha': 0.8},
                'width': 0.8
            },
            order = ['unfiltered F', 'filtered F', 'datamodel'],
            stripplot_kwargs={'alpha': 0.6, 'size': 0}
        )

    val = df_wide[df_wide['input']=='datamodel']['mean_auc_diff'].mean()

    axes[2].axhline(val, linestyle=':', label=r'$\mu_{\mathrm{datamodel}}$', color='red')
    axes[2].legend(
    loc='upper left',              # anchor the legend’s “upper left” corner
    bbox_to_anchor=(1.02, 1),      # at (x=1.02, y=1) in axes coords
    borderaxespad=0,
    title='N partitions',      
    frameon=False          # no padding between axes and legend
    )

    for ax in axes:
        ax.set_xlabel('')

    axes[0].set_ylabel('intra-partition AUC')
    axes[1].set_ylabel('inter-partition AUC')
    axes[2].set_ylabel('intra-partition AUC - inter-partition AUC')

    #axes[0].set_title('evaluation on split', pad=10, position=(0,1), fontweight='bold')

    axes[0].set_title('AUC$_1$: train and eval on same cluster', fontweight='normal', fontsize=10)
    axes[1].set_title('AUC$_2$: train and eval on diff cluster', fontweight='normal', fontsize=10)
    axes[2].set_title('AUC$_1$ - AUC$_2$',  fontweight='normal', fontsize=10)


    fig.suptitle(
        tissue,
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    

