#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example illustrating the computation of the gaze angle relative to
the horizontal (ground) plane (Figure 2).

Convention for coordinate system used here:
y-axis: temporal-to-nasal axis
x-axis: interaural axis (positive to the right)
z-axis: anti-gravity axis

Eye axes azimuth/elevation are based on Sakatani & Isa (2007) or Oommen & Stahl (2008):

Sakatani & Isa: Quantitative analysis of spontaneous saccade-like
rapid eye movements in C57BL/6 mice, Neuroscience Research 58 (2007)
324-331.

Oommen & Stahl: Eye orientation during static tilts and its relationship to spontaneous head pitch
in the laboratory mouse. Brain Res 1193: 57-66, 2008.

author: arne.f.meyer@gmail.com
"""

from __future__ import print_function

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.transform import Rotation as ssR
import click

from helpers import load_data, simple_xy_axes, set_font_axes, adjust_axes, EYE_COLORS, NICE_COLORS


def get_eye_axis_azimuth_elevation(eye_model='sakatani+isa'):
    """eye axis azimuth/elevation (Sakatani & Isa (2007) or Oommen & Stahl (2008))"""

    assert eye_model in ['oommen+stahl', 'sakatani+isa']

    if eye_model == 'oommen+stahl':
        azimuth = 64.
        elevation = 22.
    else:
        azimuth = 60.
        elevation = 30.

    return azimuth, elevation


def rotation_matrix(theta, axis='x'):
    """3D rotation matrix around on of the Cartesian axes (x/y/z)"""

    assert axis in ['x', 'y', 'z']

    rad = np.deg2rad(theta)
    c, s = np.cos(rad), np.sin(rad)

    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
    elif axis == 'y':
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])

    return R


def Rx(theta):
    return rotation_matrix(theta, 'x')


def Ry(theta):
    return rotation_matrix(theta, 'y')


def Rz(theta):
    return rotation_matrix(theta, 'z')


def compute_gaze_angle(data,
                       eye_model='sakatani+isa'):
    """horiz. gaze angle computation by simple multiplication of rotation matrixes"""

    N = len(data['timestamps'])
    angles_eye_axis = np.NaN * np.zeros((2, N))
    angles_gaze = np.NaN * np.zeros((2, N))

    # pre-compute eye-in-head rotation matrix as it does not depend on head
    # pitch/roll or eye position
    azimuth, elevation = get_eye_axis_azimuth_elevation(eye_model)

    R_eye_in_head = {}
    for eye in ['left', 'right']:
        # +- 1 for left/right eye axes
        s = 1 if eye == 'left' else -1
        R_eye_in_head[eye] = np.dot(Rz(s * azimuth), Rx(elevation))

    # the eye axis is initially aligned with the temporal-to-nasal axis (y-axis; see above)
    eye_axis = np.array([0, 1, 0])

    for j, (pitch, roll) in enumerate(zip(data['pitch'],
                                          data['roll'])):

        # head tilt relative to the horizontal (ground) plane
        R_head_in_space = np.dot(Ry(roll), Rx(pitch))

        for i, eye in enumerate(['left', 'right']):

            eye_h, eye_v = data['pupil'][eye][j, :]

            # eye position in head
            R_eye_in_orbit = np.dot(Rz(-eye_h), Rx(eye_v))

            # eye axis in space
            R_eye_in_space = np.dot(R_head_in_space, R_eye_in_head[eye])
            eye_axis_vec = np.dot(R_eye_in_space, eye_axis)

            # include current eye position
            R = np.dot(R_eye_in_space, R_eye_in_orbit)
            gaze_vec = np.dot(R, eye_axis)

            # compute angle with horizontal plane
            angles_eye_axis[i, j] = angle_with_horizontal(eye_axis_vec)
            angles_gaze[i, j] = angle_with_horizontal(gaze_vec)

    return angles_eye_axis, angles_gaze


def angle_with_horizontal(v_):
    """project onto vertical unit vector to find angle with horizontal plane"""
    return np.rad2deg(np.arcsin(np.dot(v_, [0, 0, 1]) / np.sqrt(np.sum(v_**2))))


def circmean(angles_deg, **kwargs):
    """simple wrapper around stats.circmean to handles angles in degrees"""
    return np.rad2deg(stats.circmean(np.deg2rad(angles_deg), **kwargs))


def circstd(angles_deg, **kwargs):
    """simple wrapper around stats.circstd to handles angles in degrees"""
    return np.rad2deg(stats.circstd(np.deg2rad(angles_deg), **kwargs))


def run_example():

    data = load_data()
    angles_eye_axis, angles_gaze = compute_gaze_angle(data,
                                                      eye_model='sakatani+isa')

    # ----- mean+-sd angle with horizontal plane -----
    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             sharex=True,
                             sharey=True)

    color_gaze = .6 * np.array(NICE_COLORS['orchid']) + .4 * np.array([1, 1, 1])
    color_eye_axis = 3 * [.25]

    for i, eye in enumerate(['left', 'right']):

        print("----- %s eye -----" % eye.title())

        ax = axes[i]

        for what, angles, c in [('gaze',
                                 angles_gaze[i, :],
                                 color_gaze),
                                ('eye axis',
                                 angles_eye_axis[i, :],
                                 color_eye_axis)]:

            angles = angles[~np.isnan(angles)]
            print("  %s angle = %.1f +- %.1f" % (
                what,
                circmean(angles),
                circstd(angles)))

            cnt, edges = np.histogram(angles,
                                      range=(-60, 60),
                                      bins=25)
            ax.plot(edges[:-1]+.5*(edges[1] - edges[0]), cnt/float(np.sum(cnt)), '-',
                    color=c,
                    alpha=.5+i*.5)

        ax.set_xlabel('Angle with horiz. plane (deg)')
        ax.set_ylabel('Prob.')

        ax.set_xlim(-60, 60)
        simple_xy_axes(ax)
        set_font_axes(ax)
        adjust_axes(ax)

    fig.set_size_inches(5, 2.25)
    fig.tight_layout()

    # ----- show example traces -----
    fig, axes = plt.subplots(nrows=5,
                             ncols=1,
                             sharex=True,
                             sharey=True)

    ts = data['timestamps']

    # head pitch and roll
    ax = axes[0]
    ax.set_title('Head tilt',
                 fontweight='bold')

    ax.plot(ts, data['pitch'], '-',
            color=3*[.6],
            label='pitch')
    ax.plot(ts, data['roll'], '-',
            color=3 * [.1],
            label='roll')

    ax.set_ylabel('Head pitch/roll\n(deg)',
                  labelpad=0)
    ax.legend(loc=(.9, .9),
              fontsize=6).get_frame().set_visible(0)

    for i, eye in enumerate(['left', 'right']):

        ax = axes[1 + i]
        ax.set_title('%s eye position' % eye.title(),
                     fontweight='bold')
        for j, eye_dim in enumerate(['horizontal', 'vertical']):
            ax.plot(ts, data['pupil'][eye][:, j], '-',
                    color=EYE_COLORS[eye_dim][eye],
                    label=eye_dim)

        ax.set_ylabel('Eye position\n(deg)',
                      labelpad=0)
        ax.legend(loc=(.9, .9),
                  fontsize=6).get_frame().set_visible(0)

    # gaze angle
    ax = axes[3]
    ax.set_title('Gaze angle with horiz. plane',
                 fontweight='bold')

    for i, eye in enumerate(['left', 'right']):
        ax.plot(ts, angles_gaze[i, :], '-',
                color=color_gaze,
                alpha=.5 + i*.5,
                label='%s eye' % eye.title())

    ax.set_ylabel('Angle with horiz.\n(deg)',
                  labelpad=0)
    ax.legend(loc=(.9, .9),
              fontsize=6).get_frame().set_visible(0)

    # angle of eye axis with horizontal plane
    ax = axes[4]
    ax.set_title('Eye axis angle with horiz. plane',
                 fontweight='bold')

    for i, eye in enumerate(['left', 'right']):
        ax.plot(ts, angles_eye_axis[i, :], '-',
                color=color_eye_axis,
                alpha=.5 + i*.5,
                label='%s eye' % eye.title())

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle with horiz.\n(deg)',
                  labelpad=0)

    ax.legend(loc=(.9, .9),
              fontsize=6).get_frame().set_visible(0)

    # adjust axes
    for ax in axes.flat:

        ax.set_xticks([0, 200, 400, 600])
        ax.set_yticks([-75, 0, 75])
        ax.set_xlim(-10, 610)
        ax.set_ylim(-75, 75)
        ax.spines['bottom'].set_bounds(0, 600)

        simple_xy_axes(ax)
        set_font_axes(ax)
        adjust_axes(ax)

    fig.set_size_inches(7.1, 8)
    fig.tight_layout(pad=1,
                     h_pad=2,
                     w_pad=1)

    plt.show(block=True)


if __name__ == '__main__':
    run_example()
