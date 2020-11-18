# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    ctg = CTG_features.copy()
    for feat in ctg.columns:
        for index in ctg.index:
            if not isinstance(ctg[feat][index], (float, int)):
                ctg = ctg.replace(ctg[feat][index], np.nan)
    c_ctg = {y: [x for x in ctg[y] if not pd.isna(x)] for y in ctg.columns if y != extra_feature}

    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    ctg = CTG_features.copy()
    c_cdf = rm_ext_and_nan(ctg, extra_feature)
    for feat in ctg.columns:
        for index in ctg.index:
            if not isinstance(ctg[feat][index], (float, int)):
                ctg = ctg.replace(ctg[feat][index], np.nan)
    for feat in ctg.columns:
        if feat != extra_feature:
            for index in ctg.index:
                if pd.isna(ctg[feat][index]):
                    ctg[feat][index] = np.random.choice(c_cdf[feat])
    c_cdf = rm_ext_and_nan(ctg, extra_feature)

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    d_summary = {}
    summery = c_feat.describe()
    for feat in summery.columns:
        d_summary[feat] = {}
        for statistic, key in [('min', 'min'), ('25%', 'Q1'), ('50%', 'median'), ('75%', 'Q3'), ('max', 'max')]:
            d_summary[feat][key] = summery[feat][statistic]

    # -----------------------------------------------   --------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    c_no_outlier = c_feat.copy()
    for feat in c_no_outlier.columns:
        q1 = d_summary[feat]['Q1']
        q3 = d_summary[feat]['Q3']
        iqr = q3 - q1
        for index in c_no_outlier.index:
            if c_no_outlier[feat][index] < q1 - 1.5 * iqr or c_no_outlier[feat][index] > q3 + 1.5 * iqr:
                c_no_outlier[feat][index] = np.nan

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------

    filt_feature = np.array([])
    for index in c_cdf.index:
        if c_cdf[feature][index] < thresh:
            filt_feature = np.append(filt_feature, c_cdf[feature][index])

    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    nsd_res = CTG_features.copy()
    if mode == 'mean':
        for feat in nsd_res.columns:
            nsd_res[feat] = (nsd_res[feat] - np.mean(nsd_res[feat])) / (np.max(nsd_res[feat]) - np.min(nsd_res[feat]))

    if mode == 'MinMax':
        for feat in nsd_res.columns:
            nsd_res[feat] = (nsd_res[feat] - np.min(nsd_res[feat])) / (np.max(nsd_res[feat]) - np.min(nsd_res[feat]))

    if mode == 'standard':
        for feat in nsd_res.columns:
           nsd_res[feat] = (nsd_res[feat] - np.mean(nsd_res[feat])) / np.std(nsd_res[feat])

    if flag:

        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(nsd_res[x], bins=30)
        axs[0].set_title(mode)
        axs[0].set_xlabel(x)
        axs[0].set_ylabel('Count')
        axs[1].hist(CTG_features[x], bins=30)
        axs[1].set_title('Before')
        axs[1].set_xlabel(x)
        axs[1].set_ylabel('Count')
        plt.show()

        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(nsd_res[y], bins=30)
        axs[0].set_title(mode)
        axs[0].set_xlabel(y)
        axs[0].set_ylabel('Count')
        axs[1].hist(CTG_features[y], bins=30)
        axs[1].set_title('Before')
        axs[1].set_xlabel(y)
        axs[1].set_ylabel('Count')
        plt.show()

    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
