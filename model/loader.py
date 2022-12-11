import enum
import glob
import itertools
import os
import random
import typing as tp

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


class AllowedDatasets(enum.Enum):
    PCD = 'pcd'
    PROXY = 'proxy'


def split_datasets(
    cls: Dataset,
    batch_size: int = 1,
    max_workers: int = 1,
    **kwargs
) -> tp.List[Dataset]:
    num_workers: int = max(1, min(batch_size, max_workers))
    for n in range(max_workers, 1, -1):
        if batch_size % n == 0:
            num_workers = n
            break

    batch_size = batch_size // num_workers

    return [cls(batch_size=batch_size, **kwargs) for _ in range(num_workers)]


class Hand_Gestures_Dataset(IterableDataset):
    def __init__(
        self,
        path_list: tp.List[str],
        label_map: tp.Dict[str, int],
        batch_size: int = 1,
        transforms: tp.Any = None,
        base_fps: int = 30,
        target_fps:int = 30,
        data_type: AllowedDatasets = AllowedDatasets.PCD,
    ) -> None:
        self.path_list = path_list
        self.label_map = label_map
        self.batch_size = batch_size
        self.transforms = transforms

        self.base_fps = base_fps
        self.target_fps = target_fps

        if not isinstance(data_type, AllowedDatasets):
            raise Exception('Unknown data_type for dataset')
        self.data_type = data_type

    @property
    def shuffle_path_list(self) -> tp.List[str]:
        return random.sample(self.path_list, len(self.path_list))

    def get_gesture(self, path: str) -> str:
        return os.path.basename(os.path.dirname(os.path.dirname(path)))

    def process_data(
        self,
        path: str,
        batch_idx: int,
    ):
        label = self.label_map[self.get_gesture(path)]
        with open(os.path.join(path, 'label.txt'), 'r') as label_file:
            label_start, label_finish = map(int, label_file.readline().strip().split())

        current_frame = max(0, self.base_fps - self.target_fps)

        if self.data_type == AllowedDatasets.PCD:
            paths = sorted(glob.glob(os.path.join(path, '*.pcd')))
        elif self.data_type == AllowedDatasets.PROXY:
            paths = sorted(glob.glob(os.path.join(path, '*.jpg')))

        for i, pc_path in enumerate(paths):
            current_frame += self.target_fps
            while current_frame >= self.base_fps:
                current_frame -= self.base_fps

                pc = pc_path
                if self.transforms is not None:
                    pc = self.transforms(pc, batch_idx)
                yield pc, label * (label_start <= i <= label_finish)

    def get_stream(self, data_list: tp.List[str], batch_idx: int):
        return itertools.chain.from_iterable(
            map(self.process_data, data_list, itertools.repeat(batch_idx))
        )

    def get_streams(self):
        return zip(*[self.get_stream(
            self.shuffle_path_list, batch_idx
        ) for batch_idx in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()


class MultiStreamDataLoader():
    def __init__(
        self,
        datasets: tp.List[Dataset],
        image_size: tp.Tuple[int, int],
    ) -> None:
        self.datasets = datasets
        self.image_size = image_size

    def get_stream_loaders(self):
        return zip(*[DataLoader(
            dataset, num_workers=0, batch_size=None
        ) for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            batch = list(itertools.chain(*batch_parts))
            batch_samples = torch.zeros((len(batch), 4, *self.image_size))
            batch_labels = torch.zeros(len(batch), dtype=torch.long)
            for i, sample in enumerate(batch):
                batch_samples[i] = sample[0]
                batch_labels[i] = sample[1]
            yield batch_samples, batch_labels
