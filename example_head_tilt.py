#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example script reproducing the traces shown in Figure 3 (prediction of head
tilt-related changes in eye position together with prediction errors)

author: arne.f.meyer@gmail.com
"""

from __future__ import print_function

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import click
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score

import mousecam.movement.eye as mme
from helpers import load_data, simple_xy_axes, set_font_axes, adjust_axes, EYE_COLORS


def create_regression_data(data,
                           eye='left',
                           maxlag=.1,
                           binwidth=.025):
    # prepare head tilt and eye position data for (time-lagged) regression

    ts = data['timestamps']
    pitch_roll = np.vstack((data['pitch'], data['roll'])).T
    xx, yy, valid_ind, lags = mme.create_regression_data(ts, data['pupil'][eye],
                                                         ts, pitch_roll,
                                                         maxlag=maxlag,
                                                         binwidth=binwidth,
                                                         causal=True)

    return xx, yy, lags, ts, valid_ind


def run_validation(data,
                   model='nonlinear',
                   n_folds=5,
                   verbose=True,
                   **kwargs):
    # perform cross-validation

    assert model in ['linear', 'nonlinear']

    # xx and yy contain the time-lagged covariates (i.e. pitch and roll) and eye positions,
    # respectively. valid_ind contains the time indices for all observations that were used
    # to create xx and yy and ts the time stamps of all observations.
    xx, yy, lags, ts, valid_ind = create_regression_data(data, **kwargs)
    n_samples = ts.shape[0]

    predicted = np.NaN * np.zeros((n_samples, 2))
    measured = np.NaN * np.zeros_like(predicted)
    prediction_error = np.NaN * np.zeros_like(predicted)
    r2_folds = np.zeros((n_folds, 2))

    for j, eye_dim in enumerate(['horizontal', 'vertical']):

        if verbose:
            print("  Processing eye dimension: {}".format(eye_dim))

        y = yy[:, j]  # yy[:, 0] -> horiz., yy[:, 1] -> vert.

        estimator = mme.get_estimator('MLP' if model == 'nonlinear' else 'Linear',
                                      linear_estimator='Ridge')

        # cross-validation
        cv = KFold(n_splits=n_folds,
                   shuffle=True,
                   random_state=0)

        for k, (train_ind, test_ind) in enumerate(cv.split(xx, y=y)):

            # fit model on training data
            estimator.fit(xx[train_ind, :], y[train_ind])

            # predict eye position for testing data
            y_pred = estimator.predict(xx[test_ind, :])

            # observations were randomized for cross-validation so keep
            # track of data indices
            measured[valid_ind[test_ind], j] = y[test_ind]
            predicted[valid_ind[test_ind], j] = y_pred
            prediction_error[valid_ind[test_ind], j] = y[test_ind] - y_pred

            # compute r-squared
            r2_folds[k, j] = explained_variance_score(y[test_ind],
                                                      y_pred)

            if verbose:
                print("    fold %d/%d, r-squared=%0.2f" % (k+1, n_folds, r2_folds[k, j]))

        if verbose:
            print("    r-squared = %0.2f +- %0.2f" % (
                np.mean(r2_folds[:, j]),
                np.std(r2_folds[:, j])))

    return {'measured': measured,
            'predicted': predicted,
            'prediction_error': prediction_error,
            'timestamps': ts,
            'r2_folds': r2_folds}


def prettify_axes(ax, time_interval):
    # convenience function to adjust spines, fonts etc of traces plots

    [ax.spines[s].set_visible(False) for s in ['left', 'bottom', 'top']]
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.patch.set_visible(False)

    ax.set_ylabel(u'(\u00B0)',
                  rotation=0,
                  va='center',
                  labelpad=-2)

    ax.set_xlim(time_interval[0] - .01*np.diff(time_interval),
                time_interval[1] + .01*np.diff(time_interval))
    ax.set_ylim(-35, 35)
    ax.set_xticks([])
    ax.set_yticks([-25, 25])
    ax.set_yticklabels([' -25', '+25'])
    ax.spines['right'].set_bounds(-25, 25)

    set_font_axes(ax)
    adjust_axes(ax, pad=2)


def plot_trace(ax, results_eyes,
               eye='left',
               eye_dim='horizontal',
               time_interval=(70, 80),
               time_scalebar=False):

    eye_dim_index = ['horizontal', 'vertical'].index(eye_dim)

    ts = results_eyes[eye]['timestamps']
    measured = results_eyes[eye]['measured']
    predicted = results_eyes[eye]['predicted']

    v = np.logical_and(ts >= time_interval[0],
                       ts <= time_interval[1])
    ax.plot(ts[v], measured[v, eye_dim_index], '-',
            color=EYE_COLORS[eye_dim][eye],
            clip_on=False)
    ax.plot(ts[v], predicted[v, eye_dim_index], '-',
            color=3 * [.1],
            clip_on=False)

    if time_scalebar:
        dt = round(float(np.diff(time_interval)) / 10., 1)
        x1, x2 = time_interval[1] - dt, time_interval[1]
        ax.plot([x1, x2], 2*[45], 'k-',
                lw=2,
                clip_on=False)
        ax.text(.5*(x1+x2), 50, '%g s' % dt,
                ha='center')

    prettify_axes(ax, time_interval)


def plot_error(ax, results_eyes,
               eye_dim='horizontal',
               time_interval=(70, 80)):

    eye_dim_index = ['horizontal', 'vertical'].index(eye_dim)

    for eye in ['left', 'right']:

        ts = results_eyes[eye]['timestamps']
        v = np.logical_and(ts >= time_interval[0],
                           ts <= time_interval[1])

        prediction_error = results_eyes[eye]['prediction_error'][v, eye_dim_index]
        ax.plot(ts[v], prediction_error, '-',
                color=EYE_COLORS[eye_dim][eye])

    prettify_axes(ax, time_interval)


def plot_scatter_positions(ax, results_eyes,
                           eye_dim='horizontal',
                           what='measured',
                           regression_line=True):

    eye_dim_index = ['horizontal', 'vertical'].index(eye_dim)
    x = results_eyes['left'][what][:, eye_dim_index]
    y = results_eyes['right'][what][:, eye_dim_index]
    color = .5*(np.asarray(EYE_COLORS[eye_dim]['left']) +
                np.asarray(EYE_COLORS[eye_dim]['right']))
    ax.scatter(x, y,
               s=2,
               c=color,
               edgecolors='none',
               clip_on=False)

    if regression_line:
        v = np.logical_and(~np.isnan(x), ~np.isnan(y))
        slope, intercept, r_value = stats.linregress(x[v], y[v])[:3]
        xx = np.linspace(-45, 45, 10)
        ax.plot(xx, xx*slope+intercept, '--',
                color=3*[0],
                lw=.75)
        ax.text(.8, .9, 'r=%0.2f' % r_value,
                ha='center',
                va='center',
                transform=ax.transAxes)

    ax.set_xlabel(u'Left eye pos. (\u00B0)',
                  labelpad=2)
    ax.set_ylabel(u'Right eye pos. (\u00B0)',
                  labelpad=-2)

    ax.set_xlim(-62, 62)
    ax.set_ylim(-62, 62)
    ax.set_xticks([-60, 0, 60])
    ax.set_yticks([-60, 0, 60])

    ax.spines['left'].set_bounds(-60, 60)
    ax.spines['bottom'].set_bounds(-60, 60)

    simple_xy_axes(ax)
    set_font_axes(ax)
    adjust_axes(ax, pad=2)


def run_example(model='nonlinear'):

    assert model in ['linear', 'nonlinear']

    # get measured and predicted eye positions for the two eyes
    data = load_data()

    print("Predicting eye positions from head tilt using a %s model" % model)
    results_eyes = {}
    for i, eye in enumerate(['left', 'right']):

        print("%s eye" % eye.title())
        results_eyes[eye] = run_validation(data,
                                           eye=eye,
                                           maxlag=.1,
                                           binwidth=.025,
                                           model=model,
                                           verbose=True)

    # plot measured and predicted traces
    fig = plt.figure()
    time_interval = (70, 80)  # time interval shown in Figure 3

    for j, eye_dim in enumerate(['horizontal', 'vertical']):

        for i, eye in enumerate(['left', 'right']):

            y0 = .825 - i*.125 - j*.3
            ax = fig.add_axes((.18, y0, .75, .1))

            plot_trace(ax, results_eyes,
                       eye=eye,
                       eye_dim=eye_dim,
                       time_interval=time_interval,
                       time_scalebar= j == i == 0)

            pos = ax.get_position()
            fig.text(.5*pos.x0, pos.y0+.5*pos.height,
                     '%s\n(%s eye)' % (eye_dim.title(), eye),
                     ha='center',
                     va='center',
                     fontsize=10,
                     family='Arial',
                     color=EYE_COLORS[eye_dim][eye])

    # prediction error
    for j, eye_dim in enumerate(['horizontal', 'vertical']):

        y0 = .2 - j*.125
        ax = fig.add_axes((.18, y0, .75, .1))

        plot_error(ax, results_eyes,
                   eye_dim=eye_dim,
                   time_interval=(70, 80))

        pos = ax.get_position()
        fig.text(.5*pos.x0, pos.y0+.5*pos.height,
                 'Pred. error\n(%s)' % eye_dim,
                 ha='center',
                 va='center',
                 fontsize=10,
                 family='Arial',
                 color=EYE_COLORS[eye_dim]['left'])

    # show scatter plots (left vs right eye)
    fig, axes = plt.subplots(nrows=3,
                             ncols=2)

    for j, eye_dim in enumerate(['horizontal', 'vertical']):

        for k, what in enumerate(['measured', 'predicted', 'prediction_error']):

            ax = axes[k, j]
            ax.set_title('%s (%s)' % (what.replace('_', ' ').title(), eye_dim))
            plot_scatter_positions(ax, results_eyes,
                                   eye_dim=eye_dim,
                                   what=what)

    fig.set_size_inches(4, 7)
    fig.tight_layout(pad=1,
                     h_pad=2,
                     w_pad=1)

    plt.show(block=True)


@click.command()
@click.option('--model', '-m',
              default='nonlinear',
              type=click.Choice(['linear', 'nonlinear'],
                                case_sensitive=False),
              help="Regression model for predicting eye position from head tilt ('linear' or 'nonlinear'). "
                   "Default is 'nonlinear'")
def cli(**kwargs):

    run_example(**kwargs)


if __name__ == '__main__':
    cli()
