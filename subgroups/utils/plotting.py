import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import SpectralEmbedding
import numpy as np

def plot_scatter(x, y, hue, xlabel, ylabel, title, legend_title, cmap='viridis_r', ax=None, highlight_point=None, highlight_point1 = None, highlight_point2 = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    sns.scatterplot(x=x, y=y, s=5, hue=hue, palette=cmap, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if highlight_point is not None:
        hx, hy = highlight_point
        ax.scatter(hx, hy, s=100, edgecolor='red', facecolor='none', linewidth=2, marker='o', zorder=10)

    if highlight_point1 is not None:
        hx, hy = highlight_point1
        ax.scatter(hx, hy, s=100, edgecolor='blue', facecolor='none', linewidth=2, marker='o', zorder=10)

    if highlight_point2 is not None:
        hx, hy = highlight_point2
        ax.scatter(hx, hy, s=100, edgecolor='orange', facecolor='none', linewidth=2, marker='o', zorder=10)
    # Place legend to the right
    legend = ax.legend(
        title=legend_title,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        scatterpoints=1,
        markerscale=3,
        frameon=False
    )
    ax.set_title(title)

    legend.get_title().set_verticalalignment('center')
    legend.get_title().set_horizontalalignment('left')


def get_corr(weights, index):
    corr = np.corrcoef(weights[(index)][:,((index))])
    corr = (corr+1)/2
    return corr


def get_embedding(corr, components=2):
    embedder = SpectralEmbedding(
        n_components=components,
        affinity='precomputed'
    )
    embedding = embedder.fit_transform(corr)
    return embedding

def get_sorted_corr(weights, index):
    corr = get_corr(weights, index)
    embedding = get_embedding(corr, components=1)

    # 2. Get the order of rows/columns by sorting on that embedding:
    order = np.argsort(embedding[:, 0])

    # 3. Reorder your matrix:
    sorted_corr = corr[np.ix_(order, order)]
    return sorted_corr


def plot_corr(weights, index, title, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    sorted_corr = get_sorted_corr(weights, index)
    fig.colorbar(ax.imshow(sorted_corr), ax=ax, label='Scaled Pearson Correlation Coefficient')
    ax.imshow(sorted_corr)
    ax.set_title(title)
    ax.set_ylabel('sample index')
    ax.set_xlabel('sample index')

def plot_2D_embedding(weights, index_grp_1, hue, title, legend_title, cmap='viridis_r', ax=None, s=5, show_legend=True, alpha=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    corr = get_corr(weights, index_grp_1)
    embedding = get_embedding(corr, components=2)
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=hue, palette=cmap, ax=ax, s=s, alpha=alpha)
    ax.set_xlabel('spectral embedding dim. 1')
    ax.set_ylabel('spectral embedding dim. 2')
    ax.set_title(title)
    if show_legend:
        ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    else:
        ax.legend().remove()

# chat GPT generated code
def plot_cat_continuous(
    data,
    cat_col: str,
    cont_col: str,
    order: list = None,
    box_pairs: list = None,
    test: str = "Mann-Whitney",
    text_format: str = "star",
    loc: str = "outside",
    palette=None,
    ax=None,                   # ← new parameter
    boxplot_kwargs=None,
    stripplot_kwargs=None,
    x_lower=None,
    x_upper=None,
    y_lower=None,
    y_upper=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statannotations.Annotator import Annotator
    from itertools import combinations

    # If no ax passed in, make one
    if ax is None:
        fig, ax = plt.subplots()

    boxplot_kwargs = boxplot_kwargs or {}
    stripplot_kwargs = stripplot_kwargs or dict(color="k", size=4, jitter=True, alpha=0.6)

    cats = order or sorted(data[cat_col].unique())
    pairs = box_pairs or list(combinations(cats, 2))

    sns.boxplot(
        x=cat_col, y=cont_col, data=data,
        order=cats, palette=palette, ax=ax, **boxplot_kwargs
    )
    sns.stripplot(
        x=cat_col, y=cont_col, data=data,
        order=cats, palette=palette, ax=ax, **stripplot_kwargs
    )
    if x_lower is not None:
        ax.set_xlim(x_lower, x_upper)
    if y_lower is not None:
        ax.set_ylim(y_lower, y_upper)

    annot = Annotator(
        ax, x=cat_col, y=cont_col, data=data,
        order=cats, pairs=pairs
    )
    annot.configure(test=test, text_format=text_format, loc=loc)
    annot.apply_and_annotate()


    ax.set_xlabel(cat_col)
    ax.set_ylabel(cont_col)
    return ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# chat GPT generated code
def plot_cat_proportions(
    data: pd.DataFrame,
    cat_col: str,
    hue_col: str,
    order: list = None,
    hue_order: list = None,
    palette=None,
    figsize=(6, 4),
    ax=None,
    annotate: bool = True,
):
    """
    Plots grouped barplots of the proportion of hue_col within each category of cat_col.

    Parameters
    ----------
    data : pd.DataFrame
    cat_col : str
        Column name for the x-axis categories.
    hue_col : str
        Column name for the categorical variable to compute proportions.
    order : list, optional
        Order of categories on the x-axis.
    hue_order : list, optional
        Order of hue levels.
    palette : seaborn palette, optional
    figsize : tuple, optional
        Figure size (only used if ax is None).
    ax : matplotlib Axes, optional
        Ax to draw into (if provided).
    annotate : bool, default True
        If True, annotate each bar with its percentage.
    """
    # compute counts and proportions
    counts = (
        data
        .groupby([cat_col, hue_col])
        .size()
        .reset_index(name="count")
    )
    counts["prop"] = counts.groupby(cat_col)["count"].transform(lambda x: x / x.sum())

    # create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # draw bars
    sns.barplot(
        data=counts,
        x=cat_col,
        y="prop",
        hue=hue_col,
        order=order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)

    # format y-axis as percent
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))

    # annotate bars
    if annotate:
        for bar in ax.patches:
            height = bar.get_height()
            if np.isnan(height):
                continue
            if height==0:
                continue
            ax.annotate(
                f"{height:.0%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    return ax

from matplotlib.colors import ListedColormap

def spectral_no_yellow(n_colors=256, drop_center=(0.2, 0.55)):
    """
    Returns a new colormap based on matplotlib's 'Spectral' but with the 
    central yellow band removed.

    Parameters
    ----------
    n_colors : int
        Number of samples to draw from the original colormap.
    drop_center : tuple of float
        Fractional interval (min, max) of the original [0..1] colormap
        to *drop* (i.e. where it’s the most yellow). Defaults to (0.45, 0.55).
    """
    # sample the full Spectral map
    xs = np.linspace(0, 1, n_colors)
    base = plt.cm.Spectral(xs)

    # mask out the “yellow” center
    keep = np.logical_or(xs < drop_center[0], xs > drop_center[1])
    new_colors = base[keep]

    return ListedColormap(new_colors, name="Spectral_no_yellow")