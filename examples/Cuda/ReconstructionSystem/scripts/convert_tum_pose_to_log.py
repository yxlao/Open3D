import quaternion
import numpy as np
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


if __name__ == '__main__':
    gt_dict = read_file_list('/media/wei/Data/data/cmu/a_floor/pose_file_tum.txt')
    trajectory = np.zeros((len(gt_dict), 4, 4))
    for i, ts in enumerate(gt_dict):
        pose = list(map(lambda x: float(x), gt_dict[ts]))

        t = np.array(pose[0:3])
        q = np.quaternion(pose[6], pose[3], pose[4], pose[5])
        R = quaternion.as_rotation_matrix(q)

        T = np.zeros((4, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        T[3, 3] = 1
        trajectory[i, :, :] = T

    save_traj_log('/media/wei/Data/data/cmu/a_floor/trajectory.log', trajectory)