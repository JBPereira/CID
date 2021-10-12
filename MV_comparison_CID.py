import numpy as np
import pandas as pd

from tqdm import tqdm

import os
import os.path as op
cur_path = os.getcwd()

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.use('Qt5Agg')
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import ShuffleSplit
from numpy.random import beta, gamma, normal, exponential, binomial, uniform
from sklearn.preprocessing import quantile_transform, scale

import random
from CID import CIDGmm

np.random.seed(0)
random.seed(0)


###############################################################################
# Utils
###############################################################################

def groupby_index(series):
    """
    Takes a pd.Series object with repeated indexes and converts to pd.DataFrame
    where the columns are the unique repeated indexes
    :param series: pd.Series object with repeated indexes.
    :return:
    """

    grouped_dict = {feat: series[feat].values for feat in np.unique(series.index)}

    return pd.DataFrame(grouped_dict)


######################################################################################
# Comparison Function
######################################################################################

def compare_ranking_CID(data, y, random_state=0, train_size=0.8, quant_transform=False,
                        n_splits=100, n_perm_repeats=5, data_str='MVN'):

    br_kwargs = {'tol': 1e-6, 'fit_intercept': False, 'compute_score': True}

    ###############################################################################
    # CID
    ###############################################################################

    data_ = data.copy()
    y_ = y.copy()
    if quant_transform:

        data_ = pd.DataFrame(quantile_transform(data_, n_quantiles=data_.shape[0],
                                                output_distribution='normal'),
                             columns=data.columns)
        y_ = pd.Series(quantile_transform(y_.values.reshape(-1,1), n_quantiles=data_.shape[0],
                                          output_distribution='normal').squeeze(), )
    else:
        y_ = pd.Series(y)

    cid = CIDGmm(n_bins=30, scale_data=True,
                 data_std_threshold=None,
                 empirical_mi=True, ent_pi_kwargs=br_kwargs,
                 gl_kwargs={'max_iter': 5000, 'alphas': [0.01, 0.05, 0.1, 0.3, 0.5]})

    cid.fit(X=data_, y=y_, n_samples=1)

    splitter = ShuffleSplit(train_size=train_size, test_size=1 - train_size,
                            random_state=random_state, n_splits=n_splits)

    ###############################################################################
    # Cross-validation + Permutation Importance
    ###############################################################################

    importances_permutation = []
    importances_clf = []

    with tqdm(total=n_splits) as pbar:
        for i, (train, test) in enumerate(splitter.split(data, y)):
            X_train, X_test = data.iloc[train, :], data.iloc[test, :]
            y_train, y_test = y.iloc[train].values, y.iloc[test].values

            clf = ExtraTreesRegressor(n_estimators=500, max_features=4, random_state=42)

            clf.fit(X_train, y_train)

            pis = permutation_importance(clf, X_test, y_test, n_repeats=n_perm_repeats)

            pis = np.mean(pis, axis=2)

            importances_permutation.append(np.mean(pis, axis=0))
            importances_clf.append(clf.feature_importances_)

            mean_pis = np.mean(pis, axis=0)
            cid.update_ent_pi_matrix(test, mean_pis)  # update the entropy/PI matrix with the current split

            pbar.update(1)

    ent_pi_df, mean_ent_pi, std_ent_pi = cid.predict_true_pis()

    print(mean_ent_pi)

    importances_permutation = np.array(importances_permutation)
    importances_clf = np.array(importances_clf)
    cid_imps = ent_pi_df['mean_est_pis']

    results_dict = {'perm': importances_permutation,
                    'clf': importances_clf,
                    'cid': cid_imps}

    np.save(f'results_MVN', results_dict)

    ###############################################################################
    # Ordering importance
    ###############################################################################

    cid_imps = groupby_index(cid_imps)

    median_mi = np.median(cid_imps, axis=0)
    median_perm = np.median(importances_permutation, axis=0)
    median_clf = np.median(importances_clf, axis=0)

    order_mi_imps = np.argsort(median_mi)
    mi_df = pd.DataFrame(cid_imps, columns=data.columns)

    order_imps = np.argsort(median_perm)
    permutation_df = pd.DataFrame(importances_permutation, columns=data.columns)

    tree_importance_sorted_idx = np.argsort(median_clf)

    ###############################################################################
    # Plotting
    ###############################################################################

    plt.rc('ytick', labelsize=18)
    plt.rc('xtick', labelsize=16)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))

    data_for_plot = pd.DataFrame(data=importances_clf[:, tree_importance_sorted_idx],
                                 columns=data.columns[tree_importance_sorted_idx])
    data_for_plot = data_for_plot.stack().droplevel(0)

    data_for_plot_ = pd.DataFrame(data={'x': data_for_plot.values, 'y': data_for_plot.index})

    sns.swarmplot(y='y', x='x', data=data_for_plot_, ax=ax1, orient='h',
                  color='darkblue', s=2)

    ax1.set(xlabel='', ylabel='')

    ax1.set_title('Tree\n'
                  'importances', fontsize=20, weight='bold')
    ax1.invert_yaxis()

    data_for_plot = pd.DataFrame(data=permutation_df.values[:, order_imps],
                                 columns=permutation_df.columns[order_imps])
    data_for_plot = data_for_plot.stack().droplevel(0)

    data_for_plot_ = pd.DataFrame(data={'x': data_for_plot.values, 'y': data_for_plot.index})

    sns.swarmplot(y='y', x='x', data=data_for_plot_, ax=ax2, orient='h',
                  color='darkblue', s=2)
    ax2.set_title('Permutation\n'
                  'importances', fontsize=20, weight='bold')
    ax2.set(xlabel='', ylabel='')
    ax2.invert_yaxis()

    data_for_plot = pd.DataFrame(data=mi_df.values[:, order_mi_imps],
                                 columns=mi_df.columns[order_mi_imps])
    data_for_plot = data_for_plot.stack().droplevel(0)

    data_for_plot_ = pd.DataFrame(data={'x': data_for_plot.values, 'y': data_for_plot.index})

    sns.swarmplot(y='y', x='x', data=data_for_plot_, ax=ax3, orient='h', color='darkblue', s=2)
    ax3.set_title('Covered Information\n'
                  'Disentanglement', fontsize=20, weight='bold')
    ax3.set(xlabel='', ylabel='')
    ax3.invert_yaxis()

    fig.tight_layout()

    plots_path = op.join(cur_path, 'plots')
    if not op.exists(plots_path):
        os.mkdir(plots_path)

    plt.savefig(op.join(plots_path, f'{data_str}_CID_comparison'))
    plt.show()

    print(f'mean mi: {median_mi}\n' + '*' * 10)
    print(f'mean perm: {median_perm}\n' + '*' * 10)
    print(f'mean clf: {median_clf}\n' + '*' * 10)


def mse(y_test, y_pred):
    return (y_test - y_pred) ** 2


def feature_ablation_importance(clf, clf_to_fit, X, y, train, test, loss_function='mse', random_state=0):
    metrics = {'mse': mse}

    np.random.seed(random_state)

    original_preds = clf.predict(X.iloc[test, :])

    loss = metrics[loss_function]

    original_losses = loss(y.iloc[test], original_preds)

    ablation_imps = np.zeros((len(test), X.shape[1]))

    for feat in range(X.shape[1]):
        X_ = X.copy()

        X_.drop([X.columns[feat]], axis=1, inplace=True)

        clf_to_fit.fit(X_.iloc[train, :], y.iloc[train])
        ablation_preds = clf_to_fit.predict(X_.iloc[test, :])
        ablation_losses = loss(y.loc[test], ablation_preds)
        ablation_imps[:, feat] = ablation_losses - original_losses

    return ablation_imps


def permutation_importance(clf, X_test, y_test, n_repeats=3, loss_function='mse',
                           random_state=0):
    metrics = {'mse': mse}

    np.random.seed(random_state)

    original_preds = clf.predict(X_test)

    loss = metrics[loss_function]

    original_losses = loss(y_test, original_preds)

    permutation_imps = np.zeros((*X_test.shape, n_repeats))

    for repeat in range(n_repeats):

        for feat in range(X_test.shape[1]):
            X_test_ = X_test.copy()
            X_test_.iloc[:, feat] = np.random.permutation(X_test_.iloc[:, feat])
            random_preds = clf.predict(X_test_)
            random_losses = loss(y_test, random_preds)
            permutation_imps[:, feat, repeat] = random_losses - original_losses

    return permutation_imps


if __name__ == '__main__':

    #################################################################################
    # Multivariate dataset creation
    #################################################################################

    quant_transform = True

    n_samples = 800

    X_1 = gamma(2, 2, size=n_samples)
    X_2 = beta(.5, .5, size=n_samples)
    X_3 = X_1 * X_2
    X_4 = -exponential(.2, n_samples)
    X_5 = np.sin(X_4)
    X_7 = binomial(1, 0.7, n_samples)
    X_8 = normal(-5, 1, n_samples)
    X_9 = normal(5, 1, n_samples)
    X_6 = X_7 * X_8 + (1-X_7)*X_9
    vars_ = np.array([X_1, X_2, X_3, X_4, X_5, X_6]).T
    vars_ = scale(vars_)

    y = uniform(0, 1, n_samples)

    first_piece = np.argwhere(y <= 0.15).flatten()
    second_piece = np.argwhere((y > 0.15) & (y <= 0.3)).flatten()
    third_piece = np.argwhere((y > 0.3) & (y <= 0.5)).flatten()
    fourth_piece = np.argwhere((y > 0.5) & (y <= 0.65)).flatten()
    fifth_piece = np.argwhere((y > 0.65) & (y <= 0.75)).flatten()
    sixth_piece = np.argwhere((y > 0.75) & (y <= 0.85)).flatten()
    seventh_piece = np.argwhere((y > 0.85) & (y <= 0.95)).flatten()
    eigth_piece = np.argwhere(y > 0.95).flatten()
    y[first_piece], y[second_piece], y[third_piece], \
    y[fifth_piece], y[sixth_piece], y[eigth_piece] = vars_[first_piece, 0], vars_[second_piece, 1], \
                                                      vars_[third_piece, 2], vars_[fifth_piece, 3],\
                                                      vars_[sixth_piece, 4], vars_[eigth_piece, 5]
    y[fourth_piece], y[seventh_piece] = (vars_[fourth_piece, 0] + vars_[fourth_piece, 1] + vars_[fourth_piece, 2]), \
                                        (vars_[seventh_piece, 3] + vars_[seventh_piece, 4])

    y = scale(y)
    data = pd.DataFrame(vars_,
                        columns=[f'X_{i}' for i in range(1, 7)])
    y = pd.Series(y)

    compare_ranking_CID(data, y, n_splits=200, quant_transform=True, data_str='MV_non_normal')
