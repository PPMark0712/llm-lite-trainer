import pickle
import torch
from torch.utils.data import Dataset
import random
import os


def get_bin_files(file_path):
    # 递归地获取文件夹中所有.bin后缀的完整文件路径
    bin_files = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    return bin_files


class BinaryDataset:
    """Class to handle reading from the .bin file using the .idx file offsets."""
    def __init__(self, bin_path):
        self.bin_path = bin_path
        idx_path = bin_path.replace(".bin", ".idx")
        with open(idx_path, "rb") as f:
            self.offsets = pickle.load(f)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        start_offset = self.offsets[idx]
        end_offset = self.offsets[idx + 1] if idx + 1 < len(self) else None
        with open(self.bin_path, "rb") as f:
            f.seek(start_offset)
            data = f.read() if end_offset is None else f.read(end_offset - start_offset)
        return pickle.loads(data)


class MultiFileBinaryDataset:
    """Class to handle reading from multiple .bin files using corresponding .idx files."""
    def __init__(self, file_list):
        # 创建 BinaryDataset 的实例列表
        self.datasets = [BinaryDataset(file_path) for file_path in file_list]
        # 文件长度前缀和
        self.cumulative_lengths = []
        total_length = 0
        for dataset in self.datasets:
            total_length += len(dataset)
            self.cumulative_lengths.append(total_length)

    def __len__(self):
        # 返回所有数据集的总长度
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        dataset_idx = self._find_dataset_idx(idx)
        if dataset_idx == 0:
            relative_idx = idx
        else:
            relative_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        # 返回相应 BinaryDataset 的数据
        return self.datasets[dataset_idx][relative_idx]

    def _find_dataset_idx(self, idx):
        # 找到第一个长度累计值大于idx的文件，在长度前缀和数组上二分查找
        l = 0
        r = len(self.cumulative_lengths) - 1
        ans = None
        while l <= r:
            mid = l + r >> 1
            if idx < self.cumulative_lengths[mid]:
                ans = mid
                r = mid - 1
            else:
                l = mid + 1
        return ans


class TorchMultiFileBinaryDataset(Dataset):
    """Torch-compatible Dataset class to handle multiple binary files."""
    def __init__(self, data_path, device, shuffle=False):
        self.multi_file_dataset = MultiFileBinaryDataset(get_bin_files(data_path))
        self.device = device
        self.shuffle = shuffle
        if shuffle:
            self.shuffled_indexs = [i for i in range(len(self.multi_file_dataset))]
            random.shuffle(self.shuffled_indexs)

    def __len__(self):
        return len(self.multi_file_dataset)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.shuffled_indexs[idx]
        item = self.multi_file_dataset[idx]

        if isinstance(item, tuple):
            # tuple 数据解析为 (input_ids, labels)
            input_ids = item[0]
            labels = item[1]
        else:
            # 无标签的预训练语料
            input_ids = item
            labels = item
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        return {
            "input_ids": input_ids, 
            "labels": labels,
        }