#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example script showing the different steps for eye tracking using DeepLabCut

author: arne.f.meyer@gmail.com
"""

from __future__ import print_function

import os
import os.path as op
import numpy as np
import deeplabcut
import click
import json
import glob


def find_video_files(path, pattern='*.h264'):
    # recursively find video files matching the given pattern

    video_files = []
    for root, dirs, files in os.walk(path):
        ff = glob.glob(op.join(root, pattern))
        if len(ff) > 0:
            video_files.extend(ff)

    return video_files


def exclude_video_files(video_files, files_to_exclude):
    # exclude files based on names (not including file extension)

    valid_files = []
    for vf in video_files:
        fn = op.splitext(op.split(vf)[1])[0]
        if fn not in files_to_exclude:
            valid_files.append(vf)
        else:
            print("excluding file", vf)

    return valid_files


def get_video_files(video_path, recursive, exclude,
                    file_format='h264'):
    # combine the above two functions

    files_to_exclude = list(exclude)

    if op.isdir(video_path):
        if recursive:
            video_files = find_video_files(video_path, pattern='*.' + file_format)

        else:
            video_files = glob.glob(op.join(video_path, '*.' + file_format))
        video_files = exclude_video_files(video_files, files_to_exclude)
    else:
        video_files = [video_path]

    return video_files


def convert_h264_to_mp4(video_file, output_file=None):
    # convert h264 files recorded using the "mousecam" to mp4 format

    param_file = op.splitext(video_file)[0] + '_params.json'
    with open(param_file, 'r') as f:
        dd = json.load(f)
    fps = int(dd['framerate'])

    if output_file is None:
        output_file = op.splitext(video_file)[0] + ".mp4"

    cmd = "MP4Box -fps {} -add {} {}".format(
            fps, video_file, output_file)
    os.system(cmd)

    return output_file


# ----------------------------------------------------------------------------
# Command line interpreter (CLI) entry points to call the above functions
# ----------------------------------------------------------------------------

@click.group()
def cli():
    pass


@click.command(name='convert-h264')
@click.argument('video_path')
@click.option('--recursive', '-r', is_flag=True,
              help='recursively search for h264 files')
@click.option('--exclude', '-E', multiple=True, default=[],
              help='exclude video files matching this pattern (mulitple possible)')
def cli_convert_h264(video_path, exclude):
    """convert h264 files to mp4"""

    video_files = get_video_files(video_path, recursive, exclude)

    for vf in video_files:
        convert_h264_to_mp4(vf)


cli.add_command(cli_convert_h264)


@click.command(name='create-project')
@click.argument('project_path')
@click.argument('video_paths', nargs=-1)
@click.option('--name', '-n', default='eye_video_demo',
              help='DLC project name. Default: eye_video_demo')
@click.option('--experimenter', '-e', default='experimenter',
              help='DLC experimenter name. Default: experimenter')
@click.option('--recursive', '-r', is_flag=True,
              help='Recursively search for video files')
@click.option('--format', '-f', 'format_',
              type=click.Choice(['h264', 'mp4'],
                                case_sensitive=False),
              default='mp4',
              help="File format when using the recursive mode (either 'mp4' or 'h264'). Default: mp4")
@click.option('--exclude', '-E', multiple=True, default=[],
              help='Exclude video files matching this pattern (mulitple possible)')
@click.option('--num-frames', '-n', default=20,
              help='Number of frames to manually label. Default: 20')
@click.option('--train', '-t', is_flag=True,
              help='Train network (make sure to have a GPU when using this option)')
@click.option('--analyze', '-a', is_flag=True,
              help='Extract pupil positions after training')
@click.option('--create-videos', '-c', is_flag=True,
              help='Create videos with tracking markers after pupil position extraction')
def cli_create_project(project_path,
                       video_paths,
                       name='eye_video_demo',
                       experimenter='experimenter',
                       recursive=False,
                       format_='mp4',
                       exclude=[],
                       num_frames=20,
                       train=False,
                       analyze=False,
                       create_videos=False):
    """run all steps to create a DLC project"""

    if len(video_paths) == 0:
        # use provided example
        video_files = [op.join(op.split(op.realpath(__file__))[0], 'data', 'example_eye_camera_video.mp4')]
    else:
        # get video files
        video_files = []
        for vp in list(video_paths):
            video_files.extend(get_video_files(vp, recursive, exclude,
                                               file_format=format_.lower()))

    # list all video files (and convert to mp4 if required)
    for i, vf in enumerate(video_files):
        print("found video file: %s" % vf)

        if op.splitext(vf)[1] == '.h264':
            vide_files[i] = convert_h264_to_mp4(vf)

    # create a new project
    config_path = deeplabcut.create_new_project(name, experimenter, video_files,
                                                working_directory=project_path,
                                                copy_videos=False)

    config = deeplabcut.utils.read_config(config_path)
    config['bodyparts'] = ['pupil_center', 'nasal_corner', 'temporal_corner']
    config['numframes2pick'] = num_frames
    deeplabcut.utils.write_config(config_path, config)

    # extract and label frames
    deeplabcut.extract_frames(config_path,
                              mode='automatic',
                              algo='kmeans',
                              crop=True)
    deeplabcut.label_frames(config_path)
    deeplabcut.check_labels(config_path)

    # create training dataset
    deeplabcut.create_training_dataset(config_path)

    if train:
        # train and evaluate the network
        deeplabcut.train_network(config_path)
        deeplabcut.evaluate_network(config_path)

        if analyze:
            deeplabcut.analyze_videos(config_path, video_files)

            if create_videos:
                # create a video
                deeplabcut.create_labeled_video(config_path, video_files)


cli.add_command(cli_create_project)


if __name__ == '__main__':
    cli()
