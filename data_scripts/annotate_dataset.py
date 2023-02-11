import glob
import os

import utils.utils_o3d as utils_o3d
from config import CONFIG

GESTURES_MAP = CONFIG['gestures']['gestures_set']
GESTURES_SET = [gesture[0] for gesture in GESTURES_MAP]

GESTURES_PARAMS = CONFIG['annotation']['gestures_params']

SAVE_DIR = CONFIG['directories']['datasets']['processed_dir']


def main():
    for participant in glob.glob(os.path.join(SAVE_DIR, 'G*')):
        for gesture in GESTURES_SET:
            for hand in glob.glob(os.path.join(participant, gesture, '*')):
                for trial in glob.glob(os.path.join(hand, '*')):
                    pc_paths = sorted(glob.glob(os.path.join(trial, '*.pcd')))

                    centers = []

                    for pc_path in pc_paths:
                        pc = utils_o3d.read_point_cloud(pc_path)
                        centers.append(pc.get_center()[
                                       GESTURES_PARAMS[gesture]['coord']])

                    threshold_start = (
                        centers[0] - min(centers)) * GESTURES_PARAMS[gesture]['ratio'] + min(centers)
                    threshold_finish = (
                        centers[-1] - min(centers)) * GESTURES_PARAMS[gesture]['ratio'] + min(centers)

                    for i in range(len(centers)):
                        if centers[i] < threshold_start:
                            start = i
                            break

                    for i in range(len(centers) - 1, -1, -1):
                        if centers[i] < threshold_finish:
                            finish = i
                            break

                    with open(os.path.join(trial, 'label.txt'), 'w') as f:
                        f.write(f'{start} {finish}\n')

                    break


if __name__ == '__main__':
    main()
