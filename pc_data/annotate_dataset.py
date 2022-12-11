import glob
import os

import open3d as o3d

GESTURES_SET = (
    # 'high',
    'start',
    'select',
    'swipe_right',
    'swipe_left',
)

GESTURES_PARAMS = {
    'start': {
        'coord': 2,
        'ratio': 0.2,
    },
    'select': {
        'coord': 2,
        'ratio': 0.4,
    },
    'swipe_right': {
        'coord': 2,
        'ratio': 0.4,
    },
    'swipe_left': {
        'coord': 2,
        'ratio': 0.4,
    },
}

DATA_DIR = os.path.join(
    os.path.expanduser('~'),
    'personal',
    'gestures_navigation',
    'pc_data',
    'dataset'
)


def main():
    for participant in glob.glob(os.path.join(DATA_DIR, 'G*')):
        for gesture in GESTURES_SET:
            for hand in glob.glob(os.path.join(participant, gesture, '*')):
                for trial in glob.glob(os.path.join(hand, '*')):
                    pc_paths = sorted(glob.glob(os.path.join(trial, '*.pcd')))

                    centers = []

                    for pc_path in pc_paths:
                        pc = o3d.io.read_point_cloud(pc_path)
                        centers.append(pc.get_center()[GESTURES_PARAMS[gesture]['coord']])

                    threshold_start = (centers[0] - min(centers)) * GESTURES_PARAMS[gesture]['ratio'] + min(centers)
                    threshold_finish = (centers[-1] - min(centers)) * GESTURES_PARAMS[gesture]['ratio'] + min(centers)

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
