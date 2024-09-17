import unittest
import shutil
import os, sys

sys.path.append("..")
sys.path.append("../src")

from src import Data2Set


class TestData2Set(unittest.TestCase):

    def setUp(self):
        self.test_dir = 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, 'test_file.txt'), 'w') as f:
            f.write('Test data\n' * 1000)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        d2s = Data2Set()
        self.assertEqual(d2s.columns, ['data'])
        
        d2s = Data2Set(data_column='text', feature_columns=['lang', 'source'])
        self.assertEqual(d2s.columns, ['text', 'lang', 'source'])

    def test_staging(self):
        d2s = Data2Set(feature_columns=['lang'])
        d2s.stage(self.test_dir, feature_values=['en'])
        self.assertEqual(len(d2s), 1)

    def test_build(self):
        d2s = Data2Set(feature_columns=['lang'])
        d2s.stage(self.test_dir, feature_values=['en'])
        output_dir = os.path.join(self.test_dir, 'output')
        d2s.build(output_dir, max_bytes='1KB', test_size=0.2)
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'train')))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'test')))

    def test_convert_to_bytes(self):
        d2s = Data2Set()
        self.assertEqual(d2s._convert_to_bytes('1KB'), 1024)
        self.assertEqual(d2s._convert_to_bytes('1MB'), 1024 * 1024)
        self.assertEqual(d2s._convert_to_bytes('1GB'), 1024 * 1024 * 1024)

    def test_split_chunk(self):
        d2s = Data2Set()
        chunk = "This is a test chunk"
        self.assertEqual(d2s._split_chunk(chunk, None, None), ["This is a test chunk"])
        self.assertEqual(d2s._split_chunk(chunk, " ", None), ["This is a test chunk"])
        self.assertEqual(d2s._split_chunk(chunk, None, 5), ["This ", "is a ", "test ", "chunk"])

    def test_read_data(self):
        d2s = Data2Set()
        with open(os.path.join(self.test_dir, 'test_file.txt'), 'r') as f:
            chunks = list(d2s._read_data(f, 100))
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(''.join(chunks), 'Test data\n' * 1000)

if __name__ == '__main__':
    unittest.main()