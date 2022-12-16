import glob
import os
import random

GESTURES_SET = (
    'start',
    'select',
)

# PC_DATA_DIR = os.path.join(
#     os.path.expanduser('~'),
#     'personal',
#     'gestures_dataset',
#     'HuaweiGesturesDataset',
#     'undistorted'
# )
PC_DATA_DIR = os.path.join(
    'D:\\',
    'GesturesNavigation',
    'dataset',
)

random.seed(0)

def main():
    for participant in glob.glob(os.path.join(PC_DATA_DIR, 'G*')):
        for gesture in GESTURES_SET:
            for hand in glob.glob(os.path.join(participant, gesture, '*')):
                for trial in glob.glob(os.path.join(hand, '*')):
                    label_path = os.path.join(trial, 'label.txt')
                    if not label_path:
                        continue

                    with open(label_path, 'r') as label_file:
                        label_start, label_finish = map(int, label_file.readline().strip().split())
                    rgb_paths = sorted(glob.glob(os.path.join(trial, '*.jpg')))

                    if not rgb_paths:
                        continue

                    crop_start = random.randint(int(os.path.splitext(os.path.basename(rgb_paths[0]))[0]), label_start - 1)
                    crop_end = random.randint(label_finish + 1, int(os.path.splitext(os.path.basename(rgb_paths[-1]))[0]))

                    for rgb_path in rgb_paths:
                        depth_path = os.path.join(os.path.splitext(rgb_path)[0] + '.png')
                        frame_id = int(os.path.splitext(os.path.basename(rgb_path))[0])
                        if frame_id <= crop_start or frame_id >= crop_end:
                            os.remove(rgb_path)
                            os.remove(depth_path)


if __name__ == '__main__':
    main()
