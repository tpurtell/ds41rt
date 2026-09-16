import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('exl3_package',
    Path(__file__).resolve().parents[1] / 'tools/package_v41_exl3_aot.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageDestinationTests(unittest.TestCase):
    def test_fresh_ninja_directory_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'exl3'
            package.validate_destination(output)
            for layout in ('rtx-tp1', 'rtx-tp2', 'dspark'):
                for rows in (1, 16, 80):
                    (output / layout / f'm{rows}').mkdir(parents=True)
            package.validate_destination(output)

    def test_unknown_payload_and_symlinks_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'exl3'
            nested = output / 'rtx-tp1/m16'
            nested.mkdir(parents=True)
            payload = nested / 'unrecognized.o'
            payload.write_bytes(b'keep')
            with self.assertRaises(ValueError):
                package.validate_destination(output)
            self.assertEqual(payload.read_bytes(), b'keep')
            payload.unlink()
            payload.symlink_to('missing-target')
            with self.assertRaises(ValueError):
                package.validate_destination(output)
            self.assertTrue(payload.is_symlink())

    def test_marked_package_remains_replaceable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'manifest.json').write_text(json.dumps({'schema':'ds41rt.exl3-package.v1'}))
            package.validate_destination(output)


if __name__ == '__main__':
    unittest.main()
