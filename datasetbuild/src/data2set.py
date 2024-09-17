from typing import Union, IO, Generator
import os
import pandas as pd
import pyarrow.parquet as pq

import time
import psutil
from concurrent.futures import ThreadPoolExecutor
# Test functions
def time_operation(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Operation took {end_time - start_time:.4f} seconds")
        return result
    return wrapper
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # in MB
def print_memory_usage(func):
    def wrapper(*args, **kwargs):
        print(f"Memory before: {get_memory_usage():.2f} MB")
        result = func(*args, **kwargs)
        print(f"Memory after: {get_memory_usage():.2f} MB")
        return result
    return wrapper


# Main Class
class Data2Set:
    ''' Takes in the column names / features you want in your dataset. Can be used to build a dataset for training a model or for feature engineering from a set of files.

    Methods:
    1. Staging the build with multiple directories and feature assignments.
    2. Building the dataset with parameters to:
        a. Assign the length of data in each row
        b. Shuffle and split dataset into train and test with specified ratio
        c. Set a max file output size (by creating parquet shards)
        d. Set a max row length
        e. Set a separator for splitting data in each file
    3. Reading the parquet files back into a pandas DataFrame
        
    ### Parameters
        **data_column**: The name for the column your be reading data into. Default: 'data'
        **feature_columns**: The name(s) of columns where you'll be assigning feature values. Format: ['col1', 'col2', ... , 'coln']. (default = [] == no feature columns).
    '''
    def __init__(
            self,
            data_column: str = 'data',
            feature_columns: list[str] = []
        ) -> None:
        self.columns = [data_column] + feature_columns
        self.row_config = pd.DataFrame(columns=self.columns)
        self.current_file = None
        self.current_size = 0


    def stage(self, path: str, feature_values: list[str] = []) -> None:
        ''' Stages the list of files needed to build the dataset
        and contains them in self.rowConfig.

        ### Parameters
            **path**: The full path from current path to the directory or file containing data you wish to stage for building.
            **feature_values**: List of values/labels for the data rows to be added at path. Must correspond to the feature columns created in initialization. (default = [])
        '''
        # Error handling
        assert len(feature_values) == len(self.columns)-1, f"feature_values = {len(feature_values)} and feature_columns = {len(self.columns)-1}. They must be the same shape"

        # Walk through directory and add the full_path of each file and assigned feature values to respective rows in a DataFrame.
        file_feature_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_feature_list.append([os.path.join(root, file)] + feature_values)

        # Append to row configuration list
        self.row_config = pd.concat(
            [pd.DataFrame(file_feature_list, columns=self.columns), self.row_config],
            ignore_index=True
        )


    def build(
            self,
            output_dir: str,
            max_bytes: str = "250MB",
            test_size: float = 0.15,
            random_state: int = 42,
            sep: Union[str, None] = None,
            max_row_len: Union[int, None] = None
        ) -> None:
        ''' Builds the dataset by shuffling, splitting into train and test sets, and saving as parquet files.

        ### Parameters
            **output_dir**: Directory to save the output files.
            **max_shard**: Maximum size of each shard. Must be an integer followed by KB, MB, or GB (default "250MB").
            **test_size**: Fraction of data to use for test set (default 0.2).
            **random_state**: Seed for random number generator (default 42).
            **sep**: Separator to use when splitting chunks (default None).
            **max_row_len**: Maximum length of each row (default 512).
        '''
        # Shuffle the entire dataset TODO: May want to do this after getting data because each row will still have many lines of data with the same feature values from a given file.
        shuffled_config = self.row_config.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split into train and test
        split_index = int(len(shuffled_config) * (1 - test_size))
        train_config = shuffled_config.iloc[:split_index]
        test_config = shuffled_config.iloc[split_index:]
        
        # Convert to bytes depending on last two characters of max_bytes
        max_bytes = self._convert_to_bytes(max_bytes)
        
        # Read and write data in configuration to parquet files by yielding when max file size is reached.
        commitments_test = self._commit_file_data(test_config, max_bytes, sep, max_row_len)
        self._write_commitments(commitments_test, output_dir, 'test')
        commitments_train = self._commit_file_data(train_config, max_bytes, sep, max_row_len)
        self._write_commitments(commitments_train, output_dir, 'train')


    def _write_commitments(self, commitments, output_dir, subset):
        '''
        Saves the commitments to the output directory.

        ### Parameters
            **commitments**: List of commitments to save.
            **output_dir**: Directory to save the commitments.
            **subset**: Subset of the data to save.
        '''
        os.makedirs(os.path.join(output_dir, subset), exist_ok=True)
        for i, commitment in enumerate(commitments):
            df = pd.DataFrame(commitment, columns=self.columns)
            df.to_parquet(os.path.join(output_dir, subset, f'shard_{i}.parquet'))

    
    def _commit_file_data(
            self,
            df: 'pd.DataFrame',
            max_bytes: int,
            sep: Union[str, None],
            max_row_len: Union[int, None]
        ) -> Generator[list[str], None, None]:
        ''' Commits file data to chunks based on max_size.

        ### Parameters
            **df**: DataFrame containing file paths and feature values.
            **max_size**: Maximum size of each chunk in bytes.
            **sep**: Separator to use when splitting file data.
            **max_row_len**: Maximum length of each row to be committed.

        ### Yields
            List of committed rows to write to dataset.
        '''
        committed_list = []
        file_size = 0

        # Iterate through each file to be read in self.rowConfig (each row will contain its own distinct feature values)
        for row in df.to_numpy().tolist():
            data_file_path = row[0] # e.g. '../archbuild/hexdumps/hexdump_big-plus-libraries/printf.bin'
            feature_values = row[1:]

            # Read in file data in chunks of max_size (or less)
            with open(data_file_path, 'r') as file:
                for chunk in self._read_data(file, max_bytes):
                    chunk_split = self._split_chunk(chunk, sep, max_row_len) # Split the chunk into rows of data (if applicable)
                    
                    # Iterate through each row in chunk_split
                    for chunk_row in chunk_split:
                        # Calculate the size of the current row
                        row_size = len(chunk_row)
                        # Yield current commitment when max file size is reached
                        if file_size + row_size >= max_bytes:
                            yield committed_list
                            committed_list = []
                            file_size = 0
                        
                        # Append data + feature values to current list of data to commit
                        committed_list.append([chunk_row] + feature_values)
                        file_size += row_size

        # If there's any remaining data, yield it
        if committed_list:
            yield committed_list


    def _split_chunk(self, chunk: str, sep: Union[str, None], max_row_len: Union[int, None]):
        ''' Splits a chunk of text into smaller pieces based on sep and max_row_len.

        ### Parameters
            **chunk**: The text chunk to split.
            **sep**: Separator to use when splitting chunks.
            **max_row_len**: Maximum length of each row.

        ### Returns
            List of split chunks.
        '''
        # If a separator is provided and not None, join the chunk after splitting
        if sep != None:
            chunk = ' '.join(chunk.split(sep))
        
        if max_row_len != None:
            chunkSplit = []
            # Split the chunk into rows of max_row_len
            for i in range(len(chunk) // max_row_len):
                # Extract a substring of length max_row_len and append it to chunkSplit
                chunkSplit.append(chunk[i*max_row_len:(i*max_row_len)+max_row_len])
        else:
            # If max_row_len is None, keep the entire chunk as a single row
            chunkSplit = [chunk]
        return chunkSplit
        

    def _read_data(self, f: IO[str], max_size: int):
        ''' Reads data from a file in chunks of specified size.

        ### Parameters
            **f**: File object to read from.
            **max_size**: Maximum size of each chunk.

        ### Yields
            Chunks of data from the file.
        '''
        while(True):
            chunk = f.read(max_size)
            if not chunk:
                break
            yield chunk


    def _convert_to_bytes(self, size_string):
        """
        Convert a string representation of file size to bytes.
        Supports KB, MB, and GB.
        
        ### Parameters
            **size_string**: A string like '100MB', '2GB', '500KB'
        ### Returns
            Size in bytes as an integer
        """
        size_string = size_string.upper()  # Convert to uppercase for consistency
        if size_string.endswith('KB'):
            return int(size_string[:-2]) * 1024
        elif size_string.endswith('MB'):
            return int(size_string[:-2]) * 1024 * 1024
        elif size_string.endswith('GB'):
            return int(size_string[:-2]) * 1024 * 1024 * 1024
        else:
            raise ValueError("Unsupported size unit. Use KB, MB, or GB.")
        

    def read_parquet(self, file):
        df = pq.read_pandas(file).to_pandas()
        df.to_csv('tester.csv')


    def __len__(self) -> int:
        return len(self.row_config)
    

    def __str__(self) -> str:
        return str(self.row_config)


if __name__ == '__main__':

    test = Data2Set(data_column='data', feature_columns=['endianness'])

    test.stage(path='../archbuild/hexdumps/hexdump_big-plus-libraries', feature_values=['big'])

    test.stage(path='../archbuild/hexdumps/hexdump_little-plus-libraries', feature_values=['little'])

    test.build('./dataset', sep='\n', max_row_len=1024*4)
    
    #test.read_parquet('./test'+'/file_path')