import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import math


def cohenD(df, treatment, metricName):
    """
    cohenD - calculating effect size before and after some intervention
    Parameters
    ----------
    df : dataframe
      dataframe as input
    treatment : str
      name of intervention (column)
    metricName : str
      name of covariate (column) 
    Returns
    cohenD : float
      Cohen D value returned
    """
    treated_metric = df[df[treatment] == 1][metricName]
    untreated_metric = df[df[treatment] == 0][metricName]
    # calculate denominator
    denominator = math.sqrt(((treated_metric.count()-1)*treated_metric.std()**2 + (untreated_metric.count()-1)
                            * untreated_metric.std()**2) / (treated_metric.count() + untreated_metric.count()-2))
    # if denominator is 0 divide by very small number
    if denominator == 0:
        d = abs((treated_metric.mean() - untreated_metric.mean()) / 0.000001)
    else:
        d = abs((treated_metric.mean() - untreated_metric.mean()) / denominator)
    return d
