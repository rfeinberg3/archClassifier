from typing import Any
import os
import pandas as pd

class Text2Classification:
    def __init__(self, columns) -> None:
        self.rowConfig = pd.DataFrame(columns=columns)

    def stage_rows(self, path: str, features: list[Any]) -> int:
        if os.path.isfile(path):
            self.rowConfig = pd.concat()
                

    def _file_paths(self, dirs):
        ''' Builds the list of files needed to build the dataset
        and contains them in self.file_paths
        '''
        for dir in dirs:
            self.filePaths.extend([dir+'/'+file for file in os.walk(dir).__next__()[2]])
        return self.filePaths


if __name__ == '__main__':
    os.path.join('../archbuild')
    test = Text2Classification()
    print(test.add_features('../archbuild/configs/config_ARMbig'))