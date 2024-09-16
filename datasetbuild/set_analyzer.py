import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import os
from concurrent.futures import ThreadPoolExecutor


class SetAnalyzer:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.df = self._read_parquet_dir(self.dir_path)

    def _read_parquet_file(self, file_path):
        return pq.read_table(file_path).to_pandas()

    def _read_parquet_dir(self, dir_path):
        df = pd.DataFrame()
        for root, dirs, files in os.walk(dir_path):
            with ThreadPoolExecutor() as executor:
                file_paths = [os.path.join(root, file) for file in files if file.endswith('.parquet')]
                results = executor.map(self._read_parquet_file, file_paths)
                for result in results:
                    df = pd.concat([df, result])
        return df

    # Iterate through all parquet files in the directory and visualize the data
    def visualize_data(self):
        plt.figure(figsize=(10, 6))
        # Plotting the distribution of the 'endianness' column
        self.df['endianness'].value_counts().plot(kind='bar')
        plt.title(f'Distribution of Endianness ({self.dir_path})')
        plt.xlabel('Endianness')
        plt.ylabel('Count')
        plt.show()

    def average_row_length(self):
        return self.df['data'].apply(len).mean()

if __name__ == '__main__':

    train_analyzer = SetAnalyzer('./dataset/train')
    test_analyzer = SetAnalyzer('./dataset/test')

    train_analyzer.visualize_data()
    test_analyzer.visualize_data()

    print(f'Average row length for train: {train_analyzer.average_row_length()}')
    print(f'Average row length for test: {test_analyzer.average_row_length()}') 

    '''
    train_df = load_data('./dataset/train')
    test_df = load_data('./dataset/test')
    print(train_df[0])
    print(test_df.head())
    '''