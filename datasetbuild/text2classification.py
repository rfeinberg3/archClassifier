from typing import Any
import os
import pandas as pd

class Text2Classification:
    def __init__(self, additional_columns) -> None:
        self.rowConfig = pd.DataFrame(columns=['data'].extend(additional_columns))
        self.columns = columns

    def stage_build(self, path: str, features: list[Any]) -> int:
        ''' Builds the list of files needed to build the dataset
        and contains them in self.rowConfig
        '''
        if os.path.isfile(path):
            filePath = path
            print('yo')
        else:
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    self.stage_build(full_path, features)
                else:
                    filePath = full_path
        self.rowConfig = pd.concat([pd.DataFrame([filePath].extend(features), columns=self.columns), self.rowConfig], ignore_index=True)
        print(self.rowConfig)

if __name__ == '__main__':
    os.path.join('../archbuild')
    test = Text2Classification(['data', 'endianness'])
    print(test.stage_build('../archbuild/configs/config_ARMbig', ['little']))