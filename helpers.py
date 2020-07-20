#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
some helper functions and color definitions

author: arne.f.meyer@gmail.com
"""

import numpy as np
import os.path as op

# some useful color definitions
NICE_COLORS = {'white': 3 * [255],
               'black': 3 * [0],
               'blue': [0, 120, 255],
               'orange': [255, 110, 0],
               'green': [35, 140, 45],
               'red': [200, 30, 15],
               'violet': [220, 70, 220],
               'turquoise': [60, 134, 134],
               'gray': [130, 130, 130],
               'lightgray': 3 * [150],
               'darkgray': 3 * [100],
               'yellow': [255, 215, 0],
               'cyan': [0, 255, 255],
               'dark orange': [244, 111, 22],
               'deep sky blue': [0, 173, 239],
               'deep sky blue dark': [2, 141, 212],
               'tomato': [237, 28, 36],
               'forest green': [38, 171, 73],
               'orange 2': [243, 152, 16],
               'crimson': [238, 34, 53],
               'jaguar': [35, 31, 32],
               'japanese': [59, 126, 52],
               'christi': [135, 208, 67],
               'curious blue': [2, 139, 210],
               'aluminium': [131, 135, 139],
               'buttercup': [224, 146, 47],
               'chateau green': [43, 139, 75],
               'orchid': [125, 43, 139],
               'fiord': [80, 96, 108],
               'punch': [157, 41, 51],
               'lemon': [217, 182, 17],
               'new mpl blue': [31, 119, 180],
               'new mpl red': [214, 39, 40]
               }

for k in NICE_COLORS:
    NICE_COLORS[k] = np.asarray(NICE_COLORS[k])/255.


EYE_COLORS = {'horizontal': {'left': [.75*c+.25 for c in NICE_COLORS['deep sky blue']],
                             'right': [.75*c for c in NICE_COLORS['deep sky blue dark']]},
              'vertical': {'left': [.5*c+.5 for c in NICE_COLORS['new mpl red']],
                           'right': [.75*c for c in NICE_COLORS['new mpl red']]}}


def simple_xy_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def set_font_axes(ax,
                  add_size=0,
                  size_ticks=6,
                  size_labels=8,
                  size_text=8,
                  size_title=8,
                  family='Arial'):

    if size_title is not None:
        ax.title.set_fontsize(size_title + add_size)

    if size_ticks is not None:
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=size_ticks + add_size)

    if size_labels is not None:

        ax.xaxis.label.set_fontsize(size_labels + add_size)
        ax.xaxis.label.set_fontname(family)

        ax.yaxis.label.set_fontsize(size_labels + add_size)
        ax.yaxis.label.set_fontname(family)

        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(size_labels + add_size)
            ax.zaxis.label.set_fontname(family)

    for at in ax.texts:
        at.set_fontname(family)
        if size_text is not None:
            at.set_fontsize(size_text + add_size)


def adjust_axes(ax,
                tick_length=None,
                tick_direction=True,
                ticks_inside=False,
                spine_width=0.5,
                tick_width=None,
                pad=2):

    if tick_length is None:
        tick_length = 2

    ax.tick_params(axis='both',
                   which='major',
                   length=tick_length)

    if tick_width is not None:
        ax.tick_params(axis='both',
                       which='major',
                       width=tick_width)

    if tick_direction:
        ax.tick_params(axis='both',
                       which='both',
                       direction='in' if ticks_inside else 'out')

    if pad is not None:
        ax.tick_params(axis='both',
                       which='both',
                       pad=pad)

    for s in ax.spines:
        spine = ax.spines[s]
        if spine.get_visible():
            spine.set_linewidth(spine_width)


def load_data():
    # get head tilt and eye position data as Python dictionary

    data_file = op.join(op.split(__file__)[0], 'data', 'example_freely_moving.npy')
    arr = np.load(data_file)

    # wrap into a dict
    data = {'timestamps': arr[:, 0],
            'pitch': arr[:, 1],
            'roll': arr[:, 2],
            'angular_head_velocity': {'pitch': arr[:, 3],
                                      'roll': arr[:, 4],
                                      'yaw': arr[:, 5]},
            'pupil': {'left': arr[:, [6, 7]],  # [horiz, vert]
                      'right': arr[:, [8, 9]]}}  # [horiz, vert]

    return data
