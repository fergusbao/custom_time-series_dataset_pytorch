import torch

# calculate mean and std of train set
def calculate_mean_std(df, train_size=0.8):
    data = df.to_numpy()
    data = torch.from_numpy(data)
    train_len = int(len(data) * train_size)
   
    train_data = data[:train_len]
    mean = train_data.mean(dim=0)
    std = train_data.std(dim=0)
    return mean, std 


