import torch
from torch.utils.data import Dataset, DataLoader, Subset


class TimeSeriesDataset(Dataset):
    """
    data: a Pandas DataFrame including Train, Val and Test sets
    Assume the last column in the dataset is the label
    The rest of the columns are features
    """
    def __init__(self, data, seq_len, mean=None, std=None):
        # df to ndarray
        data = data.to_numpy()
        # ndarray to tensor
        data = torch.from_numpy(data) # avoid unnecessary copy
        # normalize data
        self.data = self.normalize(data, mean, std) 
        self.seq_len = seq_len 
    
    def __len__(self):
        return len(self.data) - self.seq_len

    # z-score normalize
    def normalize(self, data, mean, std):
        if mean is None:
            return data
        else:
            data -= mean
            data /= std
            return data

    def __getitem__(self, idx):
        x = self.data[idx: idx+self.seq_len, :-1]
        y = self.data[idx+self.seq_len, -1]
        return x, y


class TimeSeriesDataLoader():
    """
    dataset: custom time-series PyTorch dataset
    train_size: 0.8 means 80% of dataset
    val_size: 0.1 means 10% of dataset
    The remaining will be used as test set
    """
    def __init__(self, dataset, train_size, val_size, batch_size, num_workers):
        self.dataset = dataset
        self.total_len = len(self.dataset)
        self.num_train = int(self.total_len*train_size)
        self.num_val = int(self.total_len*val_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def split(self):
        # split train/val/test set in order
        train_indices = [i for i in range(self.num_train)]
        val_indices = [i for i in range(self.num_train, self.num_train+self.num_val)]
        test_indices = [i for i in range(self.num_train+self.num_val, self.total_len)]

        train_set = Subset(self.dataset, train_indices)
        val_set = Subset(self.dataset, val_indices)
        test_set = Subset(self.dataset, test_indices)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader






