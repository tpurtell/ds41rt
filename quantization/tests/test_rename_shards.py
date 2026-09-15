import unittest
from rename_shards import numbered_mapping


class RenameTest(unittest.TestCase):
    def test_global_sequence_leaves_ple_last(self):
        original = [f'model-{i:05d}-of-00048.safetensors' for i in range(1, 49)]
        original += [f'ple-{layer}-{i:05d}-of-00002.safetensors' for layer in (1, 14) for i in (1, 2)]
        mapping = numbered_mapping(reversed(original))
        self.assertEqual(list(mapping.values()), [f'model-{i:05d}-of-00052.safetensors' for i in range(1, 53)])
        self.assertEqual(mapping['ple-1-00001-of-00002.safetensors'], 'model-00049-of-00052.safetensors')
        self.assertEqual(mapping['ple-14-00002-of-00002.safetensors'], 'model-00052-of-00052.safetensors')
        self.assertFalse(set(mapping) & set(mapping.values()))
        with self.assertRaises(ValueError):
            numbered_mapping(['unexpected.safetensors'])
