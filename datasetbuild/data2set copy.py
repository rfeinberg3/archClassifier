from typing import Any, Literal, Union
import os
import pandas as pd
import json
import pyarrow.parquet as pq
import io


class Data2Set:
    def __init__(
            self,
            data_column: str = 'data',
            feature_columns: list[str] = []
        ) -> None:
        self.columns = [data_column] + feature_columns
        self.rowConfig = pd.DataFrame(columns=self.columns)
        self.currentFile = None
        self.currentSize = 0
        self.fileCounter = 0


    def stage(self, path: str, feature_values: list[str] = []) -> None:
        ''' Stages the list of files needed to build the dataset
        and contains them in self.rowConfig.

        ### Parameters
            **path**: The full path from current path to the directory or file containing data you wish to stage for building.
            **feature_values**: List of values/labels for the data rows to be added at path. Must correspond to the feature columns created in initialization. ( default = [] )
        '''
        assert len(feature_values) == len(self.columns)-1, f"feature_values = {len(feature_values)} and feature_columns = {len(self.columns)-1}. They must be the same shape"

        filePaths = []
        if os.path.isfile(path):
            full_path = os.path.join(path)
            filePaths.append([full_path, feature_values])
        else:
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    self.stage(full_path, feature_values)
                else:
                    filePaths.append([full_path, feature_values])

        self.rowConfig = pd.concat(
            [pd.DataFrame(filePaths, columns=self.columns), self.rowConfig],
            ignore_index=True
        )

    def build_as_pickle(
            self,
            output_dir: str,
            max_shard: str = "250MB",
            test_size: float = 0.2,
            random_state: int = 42,
            sep: Union[str, None] = None,
            truncate: bool = True
        ) -> None:
        maxSize = int(max_shard[:-2]) * 1024 * 1024  # TODO: Convert to bytes depending on last to characters 
        
        # Shuffle the entire dataset
        # TODO: move shuffle to after sep split
        shuffled_data = self.rowConfig.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Split into train and test
        split_index = int(len(shuffled_data) * (1 - test_size))
        train_data = shuffled_data.iloc[:split_index]
        test_data = shuffled_data.iloc[split_index:]

        # Write train and test data
        # TODO: move last 3 parameters to this functions.
        train_data.to_parquet()
        self._write_pkl(output_dir, train_data, 'train', maxSize, sep, truncate)
        self._write_pkl(output_dir, test_data, 'test', maxSize, sep, truncate)


    def _write_pkl(self, output_dir, data, prefix, max_size, sep, truncate):
        os.makedirs(output_dir, exist_ok=True)
        self._open_new_file(output_dir, prefix, 'pkl')

        for _, row in self.rowConfig.iterrows():
            dataFilePath = row[self.columns[0]]
            with open(dataFilePath, 'r') as f:
                data = f.read()
            
            dataRows = [data] if sep == None else data.split(sep=sep)
            if truncate:
                dataRows.pop() if len(dataRows[-1]) != len(dataRows[0]) else None
            
            rowDict = row.to_dict()
            for line in dataRows:
                rowDict[self.columns[0]] = line  # Replace file path with actual data
                
                serializedRow = pickle.dumps(rowDict)
                rowSize = len(serializedRow)

                if self.currentSize + rowSize > max_size:
                    self._open_new_file(output_dir, prefix, 'pkl')

                self.currentFile.write(serializedRow)
                self.currentSize += rowSize

        if self.currentFile:
            self.currentFile.close()

    def _open_new_file(self, output_dir, prefix, type: Literal['pkl', 'json']):
        if self.currentFile:
            self.currentFile.close()
        self.fileCounter += 1
        newFilename = os.path.join(output_dir, f"{prefix}_shard_{self.fileCounter}.{type}")
        mode = 'wb' if type == 'pkl' else 'w' if type == 'json' else KeyError
        self.currentFile = open(newFilename, mode)
        self.currentSize = 0

    def read_dataset(self, input_dir, type: Literal['pkl', 'json']):
        for entry in os.listdir(input_dir):
            filename = os.path.join(input_dir, entry)
            mode = 'rb' if type == 'pkl' else 'r' if type == 'json' else KeyError
            with open(filename, mode) as f:
                while True:
                    try:
                        yield pickle.load(f) if type == 'pkl' else json.load(f) if type == 'json' else KeyError
                    except EOFError:
                        break

    def __len__(self) -> int:
        return len(self.rowConfig)
    
    def __str__(self) -> str:
        return str(self.rowConfig)


if __name__ == '__main__':
    
    test = Data2Set(data_column='data', feature_columns=['endianness'])

    for it in test.read_dataset('./test', type='pkl'):
        input("Go?")
        print(it)

    test.stage(path='../archbuild/hexdumps/hexdump_big-plus-libraries', feature_values=['big'])

    test.stage(path='../archbuild/hexdumps/hexdump_little-plus-libraries', feature_values=['little'])

    test.build_as_pickle('./test', sep='\n')



