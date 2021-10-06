# Test the custom time-series dataset
import numpy as np
import pandas as pd

from timeSeriesDataset import TimeSeriesDataset, TimeSeriesDataLoader



def main():
    # read data
    df = pd.read_csv('Absenteeism_at_work.csv', sep=';')
    print(df.head())
    
    # Prepare datasest and train/val/test split
    timeSeriesDataset = TimeSeriesDataset(df, seq_len=5)
    timeSeriesDataloader = TimeSeriesDataLoader(timeSeriesDataset, train_size=0.8, val_size=0.1, batch_size=64, num_workers=4)

    train_loader, val_loader, test_loader = timeSeriesDataloader.split()

    # Check the dataLoader
    # show the test set size in each batch
    for x, y in test_loader:
        print(x.shape, y.shape)

    # print the last sample's features from the last batch in test set
    print(x[-1])

    # print the labels in the last batch 
    print(y)


if __name__ == "__main__":
    main()
