import multiprocessing
from functools import reduce

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import scale
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm
from sklearn.linear_model import BayesianRidge

from CID_utils import identify_continuous_features, discretize_array

__all__ = (
    'CIDGmm',
)


class CID(metaclass=ABCMeta):

    def __init__(self, n_bins=8, scale_data=True, random_state=0, n_jobs=1,
                 empirical_mi=True, ent_pi_map_model='BR', ent_pi_kwargs=None):

        """
        Base class for Covered Information Disentanglement.
        :param n_bins: Number of bins to discretize data into
        :param scale_data: Whether to scale the data using z-tranformation
        :param random_state: random seed
        :param ent_pi_map_model: the model to learn the map between entropy values and
        permutation importance. currently, only 'BR' (Bayesian Regression) supported.
        :param ent_pi_kwargs: Keyword arguments to pass to the entropy-pi model
        """

        self.scale_data = scale_data

        self.random_state = random_state

        self.empirical_mi = empirical_mi

        self.n_bins = n_bins

        self.n_jobs = n_jobs

        self.ent_pi_map_model = ent_pi_map_model
        self.ent_pi_kwargs = ent_pi_kwargs

        self.mi_w_y = None
        self.mi_syn_df = None
        self.me_syn_df = None
        self.mi_cs_df = None
        self.me_cs_df = None
        self.ent_pi_df = None
        self.graph = None
        self.n_feats = None
        self.data = None
        self.cont_feats = None
        self.disc_feats = None

    @property
    def data(self):

        return self._data

    @data.setter
    def data(self, values):

        self._data = values

    @data.deleter
    def data(self):

        del self._data

    def identify_continuous_features(self):

        cont_feats, discrete_feats = identify_continuous_features(self.data)

        return cont_feats, discrete_feats

    def get_unique_values_table(self):

        values_table = []
        for i in range(self.n_feats):
            if i in self.cont_feats:
                values_table.append(np.linspace(
                    np.min(self.data.iloc[:, i]),
                    np.max(self.data.iloc[:, i]) + 0.001,
                    self.n_bins * 2)[1::2])

            else:
                values_table.append(np.unique(self.data.iloc[:, i]))
        return values_table

    def _get_neighbors_list(self):

        neighbors = []

        for i, row in enumerate(self.graph):

            non_zeros_inds = np.argwhere(row != 0).flatten()

            i_ind = np.argwhere(non_zeros_inds == i).flatten()[0]

            if len(non_zeros_inds) > 1:
                non_zeros_inds[[0, i_ind]] = non_zeros_inds[[i_ind, 0]]  # for easier access later on

            neighbors.append(np.array(non_zeros_inds))

        return neighbors

    @abstractmethod
    def create_graph(self, **kwargs):

        pass

    @abstractmethod
    def fit(self, X=None, y=None, sample_ids=None):
        pass

    def update_ent_pi_matrix(self, test_inds, mean_pis):
        """
        Given test_inds for the cross-validation and the mean permutation importances
        for this round, it constructs the average entropy terms to train the
        entropy-pi map model
        :param test_inds: cross-validation test indices
        :param mean_pis: mean permutation importances for this round of cross-validation
        """

        assert self.mi_w_y is not None, 'The model is not fitted yet!'

        if hasattr(self, 'data_hash_series'):
            test_inds = self.data_hash_series[test_inds]

        if self.ent_pi_df is None:
            self.ent_pi_df = pd.DataFrame([])

        data_test_ratio = len(self.mi_syn_df.values) / len(test_inds)
        mean_mi_syn = np.sum(self.mi_syn_df.loc[test_inds, :].values, axis=0) * data_test_ratio
        mean_me_syn = np.sum(self.me_syn_df.loc[test_inds, :].values, axis=0) *data_test_ratio
        mean_me_cs = np.sum(self.me_cs_df.loc[test_inds, :].values, axis=0)* data_test_ratio
        mean_mi_cs = np.sum(self.mi_cs_df.loc[test_inds, :].values, axis=0)* data_test_ratio
        ent_pi_df_ = pd.DataFrame(np.vstack([mean_pis.reshape(1, -1),
                                             mean_mi_cs.reshape(1, -1),
                                             mean_mi_syn.reshape(1, -1),
                                             mean_me_cs.reshape(1, -1),
                                             mean_me_syn.reshape(1, -1),
                                             ]),
                                  index=['mean_pis', 'mean_mics', 'mean_misyn', 'mean_mecs',
                                         'mean_mesyn'],
                                  columns=self.data.columns[:-1]).T

        ent_pi_df_baseline = pd.DataFrame(np.vstack([np.random.normal(0, 1) * 0.01 * np.min(np.abs(mean_pis)),
                                                     np.zeros((4, 1))]),
                                          index=['mean_pis', 'mean_mics', 'mean_misyn', 'mean_mecs',
                                                 'mean_mesyn'],
                                          columns=['random']).T  # to improve entopy/pi model stability
        self.ent_pi_df = pd.concat([self.ent_pi_df, ent_pi_df_, ent_pi_df_baseline], axis=0)

    def predict_true_pis(self, model=None):

        """
        Using the mean permutation importances/entropy terms matrix, builds a model
        that learns a entropy-permutation importance map and then predicts the
        permutation importance when the covered entropy is set to 0.
        :return: pi/entropy matrix with the predicted 'true' permutation importance values,
        the mean values of the matrix and standard deviation
        """

        assert self.ent_pi_df is not None, 'You need to build the entropy/pi matrix first!'

        if model is None:

            ent_pi_df, mean_df, std_df = self.fit_predict_BR()

        else:
            ent_pi_df, mean_df, std_df = self.fit_predict_general_model(model)

        return ent_pi_df, mean_df, std_df

    def fit_predict_BR(self):

        """
        Fits a Bayesian Regression model to the entropy terms and permutation importances
        and then adds a column to the ent_pi dataframe with the estimated PI values
        when the covered entropy is set to 0
        :return:
        """
        # Bayesian Regression fit

        X_train = self.ent_pi_df.loc[:, ['mean_mecs', 'mean_mics', 'mean_mesyn', 'mean_misyn']]
        X_test = self.ent_pi_df.loc[:, ['mean_mecs', 'mean_mics', 'mean_mesyn', 'mean_misyn']]
        X_test['mean_mecs'] = 0
        y_train = self.ent_pi_df['mean_pis'].values.reshape(-1, 1)
        reg = BayesianRidge(**self.ent_pi_kwargs)
        reg.fit(X_train, y_train)
        ymean, ystd = reg.predict(X_test, return_std=True)
        self.ent_pi_df['mean_est_pis'] = ymean
        ent_pi_df_ = self.ent_pi_df.copy()
        if 'random' in self.ent_pi_df.index:
            ent_pi_df_.drop(['random'], axis=0, inplace=True)

        mean_df_br = ent_pi_df_.groupby(level=0).mean()
        std_df_br = ent_pi_df_.groupby(level=0).std()
        return ent_pi_df_, mean_df_br, std_df_br

    def fit_predict_general_model(self, model, **model_kwargs):

        assert hasattr(model, 'fit'), 'Model does not have a fit method, pass a valid model.'
        assert hasattr(model, 'predict'), 'Model does not have a predict method, pass a valid model.'

        X_train = self.ent_pi_df.loc[:, ['mean_mecs', 'mean_mics', 'mean_mesyn', 'mean_misyn']]

        X_test = self.ent_pi_df.loc[:, ['mean_mecs', 'mean_mics', 'mean_mesyn', 'mean_misyn']]
        X_test['mean_mecs'] = 0

        y_train = self.ent_pi_df['mean_pis'].values.reshape(-1, 1)
        reg = model(model_kwargs)
        reg.fit(X_train, y_train)
        ymean, ystd = reg.predict(X_test, return_std=True)
        self.ent_pi_df['mean_est_pis'] = ymean
        mean_df_br = self.ent_pi_df.groupby(level=0).mean()
        std_df_br = self.ent_pi_df.groupby(level=0).std()

        return self.ent_pi_df, mean_df_br, std_df_br

    def _discretize_data_and_get_unique_values_table(self):
        """
        Data discretization
        :return: discretized data, the index of the continuous features and discrete features
        """
        data_ = self.data.copy()
        if self.cont_feats is None:
            self.cont_feats, self.discrete_feats = identify_continuous_features(data_)
        values_table = {'cont_feats': pd.DataFrame(np.zeros((self.n_bins, len(self.cont_feats))),
                                                   columns=self.cont_feats),
                        'disc_feats': dict(zip(self.discrete_feats, []))}
        for i, cont_feat in enumerate(self.cont_feats):
            data_.iloc[:, cont_feat], unique_values = \
                discretize_array(data_.iloc[:, cont_feat], n_bins=self.n_bins)
            values_table['cont_feats'].loc[:, cont_feat] = unique_values
        if len(self.discrete_feats) > 0:
            for disc_feat in self.discrete_feats:
                values_table['disc_feats'].loc[:, disc_feat] = np.unique(self.data.iloc[:, disc_feat])
        return data_, values_table

    def _get_neighbors(self, ind):

        neighbors = self.graph[ind, :].nonzero()[0]

        return neighbors

    def _remove_extreme_values(self, std_threshold=2):

        """
        Removes extreme values as these can affect the distribution fitting procedure
        :param std_threshold: rows for which any column's value exceeds std_threshold are removed
        :return:
        """

        stds = np.std(self.data.values, axis=0)
        rows_to_remove = np.hstack([np.argwhere(np.abs(self.data.values[:, i]) >
                                                std_threshold * stds[i]).flatten()
                                    for i in range(self.data.shape[1])])
        rows_to_remove = np.unique(rows_to_remove)
        self.data = self.data.drop(self.data.index[rows_to_remove])

    def _compute_entropy_terms(self, sample_ids):

        """
        Given sample ids, computes the empirical MI(X_i, Y)-H(X_i ^ Y ^ X_I) in matrix form (has the
        individual terms for each row), and assigns MI(X_i, Y)/H(X_i ^ Y ^ X_I) synergy and redundancy
        to self as attributes
        :param sample_ids:
        :return:
        """

        self.mi_w_y = self.mutual_info_with_y(sample_ids)

        mis_w_y_df = pd.DataFrame(self.mi_w_y, columns=self.data.columns[:-1])

        mis_w_y_cs_df = pd.DataFrame(mis_w_y_df.values.clip(min=0),
                                     columns=mis_w_y_df.columns, index=self.data.index)

        mis_w_y_syn_df = pd.DataFrame(np.abs(mis_w_y_df.values.clip(max=0)),
                                      columns=mis_w_y_df.columns, index=self.data.index)

        # the following is the expectation term and its equal to -H(X_i ^ Y ^ X_{i-}) + MI(X_i, Y)

        entropy_terms = self._sample_entropy_terms(sample_ids)

        multi_entro_except_mi_df = pd.DataFrame(dict(zip(entropy_terms.keys(), entropy_terms.values())))
        multi_entro_except_mi_df.columns = self.data.columns[:-1]
        multi_entro_except_mi_df /= len(sample_ids)
        multi_entro_syn_df = \
            pd.DataFrame(np.abs((mis_w_y_df - multi_entro_except_mi_df).values.clip(max=0)),
                         columns=multi_entro_except_mi_df.columns, index=self.data.index)

        multi_entro_cs_df = pd.DataFrame((mis_w_y_df - multi_entro_except_mi_df).values.clip(min=0),
                                         columns=multi_entro_except_mi_df.columns, index=self.data.index)

        self.mi_syn_df = mis_w_y_syn_df
        self.me_syn_df = multi_entro_syn_df
        self.mi_cs_df = mis_w_y_cs_df
        self.me_cs_df = multi_entro_cs_df

    def empirical_mutual_info_w_y(self):

        emp_mi = [mutual_info_score(self.data.loc[:, feat], self.data.iloc[:, -1])
                  for feat in self.data.columns[:-1]]

        return emp_mi

    @abstractmethod
    def mutual_info_with_y(self, sample_ids):

        pass

    @abstractmethod
    def _compute_covered_info(self, feat_ind, sample_ids):

        pass

    def _sample_entropy_terms(self, sample_ids):
        """
        Computes the empirical expected term in the CID value
        :param sample_ids: ids of the sampled rows
        :return:
        """

        n_feats = self.data.shape[1]
        
        print('\nSampling covered entropy terms\n')

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_covered_info)(feat_ind, sample_ids)
            for feat_ind in tqdm(range(n_feats - 1)))

        results = dict(results)

        return results

    @staticmethod
    def convert_n_samples(n_samples, n_instances):

        if isinstance(n_samples, int):

            n_samples = n_samples

        elif isinstance(n_samples, float):

            n_samples = int(n_samples * n_instances)

        return n_samples

    @staticmethod
    def select_data_by_id(data, ids):

        if isinstance(data, pd.DataFrame):
            if isinstance(ids, pd.Index):
                data = data.loc[ids, :].values
            else:
                data = data.iloc[ids, :].values
        else:
            data = data[ids, :]

        return data

    @staticmethod
    def select_data_cols(data, cols):

        if isinstance(data, pd.DataFrame):
            if isinstance(cols[0], int):
                data = data.iloc[:, cols].values
            else:
                data = data.loc[:, cols].values
        else:
            data = data[:, cols]

        return data


class CIDGmm(CID):

    def __init__(self, n_bins=8, scale_data=False,
                 threshold_precision=0.08, random_state=0, data_std_threshold=None,
                 empirical_mi=False, covered_n_jobs=-2, cov_threshold=0.05,
                 ent_pi_map_model='BR', ent_pi_kwargs=None, gl_kwargs=None):

        """
        Initializes CID using Gaussian Markov Model.
        :param n_bins: number of bins to discretize data.
        Pass int or array if different number of bins is desired
        :param threshold_precision: value below which to consider entries in the
        estimated precision as 0. Precision entries that are 0 encode independence
        between the features.
        :param empirical_mi: whether to use non-parametric measure of mutual information with output (False if not)
        :param cov_threshold: values in the graphical lasso inferred covariance below cov_threshold will be clipped to 0
        :param data_std_threshold: All values np.abs(x_i)>std * data_std_threshold will be excluded.
        This can help getting a more accurate approximation of the density function by
        removing outliers.
         :param ent_pi_map_model: the model to learn the map between entropy values and
        permutation importance. currently, only 'BR' (Bayesian Regression) and
        'GAM' (Generalized Additive Models) supported.
        :param ent_pi_kwargs: Keyword arguments to pass to the entropy-pi model
        :param gl_kwargs. arguments to pass to graphical lasso. See sklearn.covariance for more info
        """

        self.threshold_precision = threshold_precision

        self.precision = None
        self.mean = None
        self.nu = None
        self.empirical_mi = empirical_mi
        self.covered_n_jobs = covered_n_jobs
        self.data_std_threshold = data_std_threshold
        self.gl_kwargs = gl_kwargs
        self.deltas = None
        self.values_table = None

        self.random_state = random_state

        self.pi_entropy_df = None
        self.true_imp_df = None

        CID.__init__(self, n_bins=n_bins, scale_data=scale_data,
                     ent_pi_map_model=ent_pi_map_model, ent_pi_kwargs=ent_pi_kwargs)

    def create_graph(self, kwargs):

        self.compute_gaussian_cov(**kwargs)

        self.graph = np.array(np.abs(self.precision) > 0.01).astype(int)

    def compute_n_neighbors(self):

        n_neighbors = [len(nns) for nns in self.neighbors]

        return n_neighbors

    def compute_gaussian_cov(self, **kwargs):

        """
        Graphical LASSO for estimating the network graph, covariance and precision
        :param kwself. Check sklearn.GraphicalLassoCV for the parameters
        :return:
        """
        n_proc = multiprocessing.cpu_count()
        graph_lasso = GraphicalLassoCV(**kwargs, n_jobs=-2)

        graph_lasso.fit(self.data)

        estimated_mean = graph_lasso.location_
        estimated_precision = graph_lasso.precision_
        self.mean = estimated_mean
        self.precision = estimated_precision
        # self.precision[np.abs(self.precision) < 0.005] = 0
        self.nu = np.matmul(self.precision, self.mean)
        self.covariance = np.linalg.pinv(self.precision)

    def mutual_info_with_y(self, sample_ids):

        """
        Computes the empirical mutual info for each variable with the output.
        :param sample_ids: The ids of the sampled rows
        :return: MI(x_ij, y) for all i in (1,len(sample_ids)), j in (1,N_features)
        """

        data = self.select_data_by_id(self.data, sample_ids)

        # Univariate normal probabilities

        p_xs = np.array([mvn.pdf(data[:, i], mean=self.mean[i],
                                 cov=self.covariance[i, i])
                         for i in range(data.shape[1])])

        # (X_i, Y) normal probabilities

        p_x_y = np.array([mvn.pdf(data[:, [i, -1]], mean=self.mean[[i, -1]],
                                  cov=self.covariance[np.ix_([i, -1], [i, -1])])
                          for i in range(data.shape[1] - 1)])

        # Matrix of mutual information with Y terms

        MI = np.log(p_x_y / (p_xs[:-1] * p_xs[-1]))
        MI /= len(sample_ids)  # sum(MI, axis=0) is an empirical expectation

        return MI.T

    def _pre_process_data(self):

        # Remove extreme values

        if self.data_std_threshold is not None:
            self._remove_extreme_values(self.data_std_threshold)

        # Discretize data

        data, values_table = self._discretize_data_and_get_unique_values_table()

        # Scale the data to have 0-mean unit variance after the discretization

        # Scale the data

        if self.scale_data:
            from sklearn.preprocessing import StandardScaler as Scaler
            scaler = Scaler()
            data.iloc[:, self.cont_feats] = \
                pd.DataFrame(scaler.fit_transform(data.values[:, self.cont_feats]),
                             columns=self.data.columns)
            self.data = data
            values_table['cont_feats'] = scaler.transform(values_table['cont_feats'])
            cont_table = {self.cont_feats[i]: values_table['cont_feats'][:, i]
                              for i in range(len(self.cont_feats))}
            disc_table = dict(zip(self.discrete_feats, values_table['disc_feats']))
            cont_table.update(disc_table)
            self.values_table = cont_table

    def fit(self, X=None, y=None, n_samples=0.5):

        """
        Computes the CID value for each feature
        :param X: data
        :param y: target values
        :param n_samples: Number of samples to use when computing the empirical
        expectation term in the CID values
        :return: covered information for each feature
        """

        if isinstance(X, pd.DataFrame):
            data = pd.concat([X, y], axis=1)
        else:
            try:
                data = np.hstack([X, y])
            except:
                Exception('Please pass a valid format of X (pd.DataFrame or np.array/np.matrix)')

        self.data = pd.DataFrame(data)

        # Identify continuous and discrete features

        self.n_feats = self.data.shape[1]

        self._pre_process_data()

        self.deltas = [self.values_table[i][1] - self.values_table[i][0]
                       for i in range(self.n_feats)]

        if self.graph is None:
            self.create_graph(self.gl_kwargs)

        self.neighbors = self._get_neighbors_list()

        M = self.data.shape[0]

        sample_ids = np.random.choice(np.arange(M),
                                      int(n_samples * M), replace=False)
        self.entropy_y = np.log(mvn.pdf(data.values[:, -1], mean=self.mean[-1],
                                   cov=self.covariance[-1, -1])) / M

        self._compute_joint_y_pot_tensor()  # precompute the F matrix (potentials of clique c=(X_i, Y)) for each X_i

        self._compute_entropy_terms(sample_ids)

    def map_value_to_discrete_pos(self, feat_ind, value):
        if feat_ind in self.cont_feats:
            value_pos = int(
                (value - self.values_table[feat_ind][0]) / self.deltas[feat_ind] + 0.001)
        else:  # lookup the value position
            value_pos = np.argwhere(self.values_table[feat_ind] == value).flatten()[0]
        return value_pos

    def _compute_covered_info(self, feat_ind, sample_ids):

        """
        Computes expected value in covered info term.
        D is the array of potentials between feat_indure x_i and its neighbors excluding y
        E is the array of potentials between y and its neighbors excluding x_i
        F is the matrix of potentials between y and x_i for all values y, x_i
        :param feat_ind: index of feat_indure x_i
        :param sample_ids: row ids of the sample
        :return: Covered info numerator term for this sample
        """

        y = self.n_feats - 1

        result = []

        if self.graph[feat_ind, -1] == 1:  # if Y is neighbor of X_I then the values are non-zero

            for id in sample_ids:

                x_sample = self.data.iloc[id, :].values

                x_value_pos = self.map_value_to_discrete_pos(feat_ind, x_sample[feat_ind])
                y_value_pos = int((x_sample[-1] - self.values_table[y][0]) / self.deltas[-1] + 0.001)

                F = self.y_pot_tensor[feat_ind]

                first_term = np.log(F[x_value_pos, y_value_pos])

                D = self.compute_xi_exclude_xj_pot_array(i_pos=feat_ind, j_pos=-1,
                                                         x_sample=x_sample)

                E = self.compute_xi_exclude_xj_pot_array(i_pos=-1, j_pos=feat_ind,
                                                         x_sample=x_sample)

                numerator = reduce(np.matmul, [D.T, F, E])

                f_Y = F[:, y_value_pos]
                f_X_i = F[x_value_pos, :].T

                demon_1 = np.matmul(D[np.newaxis, :], f_Y)[0]
                demon_2 = np.matmul(E[np.newaxis, :], f_X_i)[0]

                final_term = first_term + np.log(numerator / (demon_1 * demon_2))

                result.append(final_term)

        else:

            result = np.zeros((len(sample_ids, )))

        return feat_ind, result

    def _compute_joint_y_pot_tensor(self):

        """
        Computes a potential tensor for each feature between each value of the feature
        and each value of y, yielding a matrix (n_feats x n_bins x n_bins) where the
        feature values vary over rows and the y values vary over columns
        :return: y_pot_tensor
        """

        y_pot_tensor = []
        y = self.n_feats - 1

        for feat in range(self.data.shape[1] - 1):
            feat_y_pair_values = np.array(np.meshgrid(self.values_table[y], self.values_table[feat])).reshape(2, -1)

            exponent = -0.5 * np.prod(feat_y_pair_values, axis=0) * self.precision[feat, -1]

            y_pot_values = np.exp(exponent.reshape(len(self.values_table[feat]),
                                                   len(self.values_table[y])))

            y_pot_tensor.append(y_pot_values)

        self.y_pot_tensor = y_pot_tensor

    def compute_xi_exclude_xj_pot_array(self, i_pos, j_pos, x_sample):

        """
        Computes the potential of feature x_i while excluding the
        potential terms that involve feature x_j
        :param i_pos: feature index whose potentials are to be computed
        :param j_pos: feature index whose potentials are not to be included
        :param x_sample: sampled data row
        :return:
        """

        nn = self.neighbors[i_pos]
        if j_pos == -1:
            j_pos = len(x_sample) - 1
        if i_pos == -1:
            i_pos = len(x_sample) - 1
        nn = nn[np.logical_and(nn != j_pos, nn != i_pos)]

        x_i_values = self.values_table[i_pos]

        x_i_only_pot = self.compute_single_var_pot_exponent(i_pos, x_i_values)

        cross_pot_array = -0.5 * np.sum(np.outer(self.precision[i_pos, nn] * x_sample[nn], x_i_values), axis=0)

        return np.exp(x_i_only_pot + cross_pot_array)

    def compute_single_var_pot_exponent(self, i_pos, x_i_values):

        """
        Computes the potential exponent that involves only feature x_i
        :param i_pos: index of the feature
        :param x_i_values: values of the feature x_i for the sampled rows
        :return:
        """

        x_i_only_pot = -0.5 * (x_i_values ** 2 * self.precision[i_pos, i_pos] -
                               2 * x_i_values * self.nu[i_pos])

        return x_i_only_pot
