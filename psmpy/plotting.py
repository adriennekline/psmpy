from psmpy import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import math
from psmpy.functions import cohenD
import seaborn as sns
sns.set(rc={'figure.figsize': (10, 8)}, font_scale=1.3)


def plot_match(dftreat, dfcontrol, matched_entity='propensity_logit', Title='Side by side matched controls', Ylabel='Number of patients', Xlabel='propensity logit', names=['treatment', 'control'], colors=['#E69F00', '#56B4E9']):
    """
    plot_match -- plot the logit or propensity scores
    Parameters
    ----------
    dftreat : dataframe
       dataframe from the treatment group
    dfcontrol : dataframe
       dataframe from the control group   
    matched_entity : str
       entity want to match on 'propensity_logit' (default) or 
    Title : str
       title for plot 
    Ylabel : str
       label for y axis
    Xlabel : str
       label for x axis
    names : list
        2 element list of strings for what to call each group ['treatment','control'] (default)
    colors : list
        2 element list of strings for colors ['#E69F00', '#56B4E9'] (default)
    Returns
    plot : image
    """
    x1 = dftreat[matched_entity]
    x2 = dfcontrol[matched_entity]
    # Assign colors for each airline and the names
    # colors = ['#E69F00', '#56B4E9']
    names = names
    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.hist([x1, x2],
             color=colors, label=names)
    # Plot formatting
    plt.legend()
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)


def effect_size_plot(df_original, df_matched, treatment='treatment', title='Standardized Mean differences accross covariates before and after matching'):
    """
    knn_matched -- Match data using k-nn algorithm
    Parameters
    ----------
    matcher : str
       string that will used to match - propensity score or proppensity logit
    replacement : bool
       Want to match with or without replacement
    caliper : float
       caliper multiplier for allowable matching
    Returns
    balanced_match : pd.DataFrame
        DataFrame with column with matched ID based on k-NN algorithm
    """
    df_original = df_original.astype(float)
    df_matched = df_matched.astype(float)
    df_matched_notreatmentlabel = df_matched.drop(labels=[treatment], axis=1)
    cols = df_matched_notreatmentlabel.columns
    data = []
    for cl in cols:
        try:
            data.append([cl, 'before', cohenD(df_original, treatment, cl)])
        except:
            data.append([cl, 'before', 0])
        try:
            data.append([cl, 'after', cohenD(df_matched, treatment, cl)])
        except:
            data.append([cl, 'after', 0])
    res = pd.DataFrame(data, columns=['variable', 'matching', 'effect_size'])
    sns.set_style("white")
    sn_plot = sns.barplot(data=res, y='variable',
                          x='effect_size', hue='matching', orient='h')
    sn_plot.set(title=title)
