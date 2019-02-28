#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
The Kinect provides the color and depth images in an un-synchronized way. This means that the set of time stamps from the color images do not intersect with those of the depth images. Therefore, we need some way of associating color images to depth images.

For this purpose, you can use the ''preprocess_tum.py'' script. It reads the time stamps from the rgb.txt file and the depth.txt file, and joins them by finding the best matches.
"""

import argparse
import sys
import os
import numpy as np
import quaternion

from process_traj_log import save_traj_log

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    lst = [[v.strip()
            for v in line.split(" ") if v.strip() != ""
            ]
           for line in lines if len(line) > 0 and line[0] != "#"
           ]
    lst = [(float(l[0]), l[1:]) for l in lst if len(l) > 1]
    return dict(lst)


def associate(first_dict, second_dict, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = list(first_dict.keys())
    second_keys = list(second_dict.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def generate_data_association(dataset_path, filename,
                              first_dict, second_dict, matches,
                              first_only):
    with open(dataset_path + '/' + filename, 'w') as fout:
        if first_only:
            for a, b in matches:
                fout.write('{} {}\n'.format(a, " ".join(first_dict[a])))
        else:
            for a, b in matches:
                fout.write('{} {}\n'.format(
                    " ".join(first_dict[a]),
                    " ".join(second_dict[b])))


if __name__ == '__main__':
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes depth, rgb, groundtruth,
     and generate data association with ground truth trajectory
    ''')
    parser.add_argument('--path', type=str,
                        default='/home/wei/Work/data/tum')
    parser.add_argument('--first_only',
                        help='only output associated lines from first file',
                        action='store_true')
    parser.add_argument('--offset',
                        help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.02)',
                        default=0.02)
    args = parser.parse_args()

    dataset_paths = os.listdir(args.path)
    for dataset_path in dataset_paths:
        print('Processing dataset {}'.format(dataset_path))
        dataset_absolute_path = args.path + '/' + dataset_path

        depth_dict = read_file_list(dataset_absolute_path + '/depth.txt')
        rgb_dict = read_file_list(dataset_absolute_path + '/rgb.txt')
        # gt_dict = read_file_list(dataset_absolute_path + '/groundtruth.txt')

        matched_depth_and_rgb = associate(depth_dict, rgb_dict,
                                          float(args.offset),
                                          float(args.max_difference))
        generate_data_association(dataset_absolute_path,
                                  'depth_rgb_association.txt',
                                  depth_dict, rgb_dict, matched_depth_and_rgb,
                                  args.first_only)
        #
        # matched_depth_and_gt = associate(depth_dict, gt_dict,
        #                                  float(args.offset),
        #                                  float(args.max_difference))
        #
        # # (depth, rgb), (gt)
        # matches = []
        # gts = []
        # for a, b in matched_depth_and_rgb:
        #     associated_detph_and_gt = \
        #         list(filter(lambda x: x[0] == a, matched_depth_and_gt))
        #     if len(associated_detph_and_gt) > 0:
        #         matches.append((a, b))
        #         gts.append(associated_detph_and_gt[0][1])
        #
        # generate_data_association(dataset_absolute_path, 'data_association.txt',
        #                           depth_dict, rgb_dict, matches,
        #                           args.first_only)
        #
        # trajectory = np.zeros((len(gts), 4, 4))
        # for i, gt in enumerate(gts):
        #     pose = list(map(lambda x: float(x), gt_dict[gt]))
        #     t = np.array(pose[0:3])
        #     q = np.quaternion(pose[6], pose[3], pose[4], pose[5])
        #     R = quaternion.as_rotation_matrix(q)
        #
        #     T = np.zeros((4, 4))
        #     T[0:3, 0:3] = R
        #     T[0:3, 3] = t
        #     T[3, 3] = 1
        #     trajectory[i, :, :] = T
        #
        # save_traj_log(dataset_absolute_path + '/trajectory.log', trajectory)
        #
        # print('({} {}->{}) {} -> {}'.format(
        #     len(depth_dict), len(rgb_dict), len(matched_depth_and_rgb),
        #     len(gt_dict), len(gts)))
