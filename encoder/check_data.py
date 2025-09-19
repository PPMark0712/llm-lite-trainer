# 读取数据集的实例
import pickle

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


if __name__ == "__main__":
    import glob
    file_list = glob.glob("/data1/yyz/projects/forget/train/dataset/alpaca/*.bin")
    for fn in file_list:
        dataset = BinaryDataset(fn)
        for item in dataset:
            if len(item[0]) != 4096 or len(item[1]) != 4096:
                print(len(item[0]), len(item[1]))
