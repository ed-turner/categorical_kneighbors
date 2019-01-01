from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from .distances import p_norm_distance, cosine_distance


class CategoricalKNNBase:

    def __init__(self, neighbors=3, cat_col=None, distance=2.0, n_jobs=None):

        if isinstance(distance, int) | isinstance(distance, float):
            self.distance_func = lambda mat_A, mat_B: p_norm_distance(mat_A, mat_B, distance)
        elif distance == 'cosine':
            self.distance_func = cosine_distance
        else:
            raise ValueError("The wrong distance measure is passed.  Make sure it is cosine, or a numeric for the " +
            "p-norm")

        self.cat_col = cat_col
        self.neighs = neighbors
        self.n_jobs = n_jobs

        return None

    def _fit_cat_val(self, cat_val, df_train, df_test):

        distance_funct = self.distance_func

        train_indices = df_train[self.cat_col] == cat_val
        test_indices = df_test[self.cat_col] == cat_val

        train_test_kernel = distance_funct(df_test.loc[test_indices, :].drop(self.cat_col, axis=1).values,
                                           df_train.loc[train_indices, :].drop(self.cat_col, axis=1).values)

        closest_train_test = np.argsort(train_test_kernel, axis=1)[:, :self.neighs]

        return {'train_indices': train_indices, 'test_indices': test_indices, 'closest_preds': closest_train_test}

    def _fit(self, df_train, df_test):

        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)

        assert df_train.shape[1] == df_test.shape[1]

        cat_vals = df_train[self.cat_col].unique()

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            exec_dict = {cat_val: executor.submit(self._fit_cat_val, cat_val, df_train, df_test)
                         for cat_val in cat_vals}

        res_dict = {cat_val: exec_dict[cat_val].result() for cat_val in cat_vals}

        return res_dict

    def _fit_predict(self, df_train, df_test, y):

        assert df_train.shape[0] == y.shape[0]

        res_dict = self._fit(df_train, df_test)

        test_pred = np.zeros((df_test.shape[0],))

        for key in res_dict.keys():

            train_indices = res_dict[key]['train_indices']
            test_indices = res_dict[key]['test_indices']
            closet_indices = res_dict[key]['closest_preds']

            test_pred[test_indices] = y[train_indices][closet_indices].mean(axis=1)

        return test_pred


class CategoricalKNNRegressor(CategoricalKNNBase):

    def __init__(self, **kwargs):

        CategoricalKNNBase.__init__(**kwargs)

    def fit_predict(self, df_train, df_test, y):
        return self._fit_predict(df_train, df_test, y)


class CategoricalKNNClassifier(CategoricalKNNBase):

    def __init__(self, **kwargs):

        CategoricalKNNBase.__init__(**kwargs)

    def fit_predict(self, df_train, df_test, y):

        if 2 < np.unique(y).shape[0]:

            y_one_hot_encode = np.zeros((y.shape[0], int(np.max(y)) + 1))

            y_one_hot_encode[np.arange(y.shape[0]), y] = 1

            probs = self._fit_predict(df_train, df_test, y_one_hot_encode)

            probs = np.dot(np.diag(probs.sum(axis=1) ** -1.0), probs)

            return np.argmax(probs, axis=1)
        else:
            probs = self._fit_predict(df_train, df_test, y)

            return np.round(probs)
