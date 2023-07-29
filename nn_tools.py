#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/07/29/the-speculative-neighbor/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

import numpy as np
import statsmodels.api as sm


def get_trajectory(x, k):
    """
    Compute the trajectory matrix of a matrix given a memory parameter

    :param x:           array of data
    :param k:           scalar, memory order
    :return:            the trajectory matrix
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if k == 1:
        return x
    elif k > 1:
        y = np.zeros([x.shape[0] - k + 1, k * x.shape[1]])
        for i in range(x.shape[0] - k + 1):
            y[i, :] = np.hstack(x[i:(i + k), :])
        return y
    else:
        raise ValueError('In function get_trajectory the memory order must be >= 1')


def get_nn(x, p=2, alpha=0.8, step=1):
    """
    Compute the nearest neighbors to the last value

    :param x:           array of data, vector or matrix
    :param p:           scalar, distance exponent (-1 or 1 or 2) for the NN; -1 is plain matching
    :param alpha:       scalar, between 0 and 1, % of NN to retain from sample
    :param step:        scalar, steps to subsample NN
    :return:            the indices in the input matrix with the nearest neighbors
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    # Extract the last row from the trajectory matrix
    x_last = x[-1, :]
    # Compute all the distances, sort and extract the indices
    if p < 0:
        similarity_score = np.sum(np.logical_and(x[:-1, :], x_last), axis=1)
        winner_pos = np.argwhere(similarity_score >= -p).flatten()
        if len(winner_pos) == 0:
            return None
        _aux = np.argsort(similarity_score[winner_pos])
        isd = winner_pos[_aux][::-1]
    else:
        if p == 0:
            dxt = np.sum(np.abs(x[:-1, :] - x_last), axis=1)
        else:
            dxt = np.sum((np.abs(x[:-1, :] - x_last) ** p), axis=1) ** (1 / p)
        isd = dxt.argsort()
        isd = isd[np.where(isd < len(x))]
    isd = isd[:int(len(isd) * alpha) + (1 if p < 0 else 0)]
    # Subsample?
    if step > 1:
        ss = np.arange(0, len(isd), step)
        isd = isd[ss]
    # Done - note that the index of the closest NN is the first in isd!
    return isd


def get_nn_forecast(x, isd, beta=0.2, nn_type=None):
    """
    Compute a modified NN by averaging around/forward/backward of each NN point

    :param x:           vector of target variables
    :param isd:         the indices of the nearest neighbors to account for
    :param beta:        scalar, between 0 and 1, % of observations to retain around each NN point
    :param nn_type:     string, the type of forecast to produce. If None, the mean of nn is returned
    :return:            the array of individual NN forecasts and their mean
    """
    isd = isd[isd < x.shape[0]]
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    z = x[isd, :]

    if nn_type is None:
        return z.mean(axis=0)

    f = np.zeros(z.shape)  # this is the modified h-step ahead data
    for i in range(isd.shape[0]):
        ix = isd[i]
        if nn_type == 'center':
            i_min = max(0, int(ix - len(x)*beta))
            i_max = min(int(ix + x.shape[0] * beta), x.shape[0])
        elif nn_type == 'forward':
            i_min = ix
            i_max = min(int(ix + x.shape[0] * beta), x.shape[0])
        elif nn_type == 'backward':
            i_min = max(0, int(ix - len(x)*beta))
            i_max = ix
        elif nn_type == 'regress':
            i_min = ix
            i_max = min(int(ix + x.shape[0] * beta), x.shape[0])
        else:
            raise ValueError('In <get_nn_forecast> wrong value passed in parameter <nn_type>.')
        if nn_type != 'regress':
            f[i] = np.mean(x[i_min:i_max, :], axis=0)
        elif nn_type == 'regress':
            xi = x[i_min:i_max, :]
            if len(xi) > 3:
                w = sm.add_constant(xi[:-1, :])
                y = xi[1:, -1]
                out = sm.OLS(y, w).fit()
                f[i] = out.params[0] + np.sum(out.params[1:] * x[-1, :])
            else:
                f[i] = np.mean(xi, axis=0)
    return f.mean(axis=0)
