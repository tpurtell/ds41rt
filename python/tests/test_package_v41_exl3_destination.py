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
    def test_paired_boundary_contract_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'tp4-rank0/m80'
            directory.mkdir(parents=True)
            meta = dict(capacity=80, intermediate=640, experts=384, top_k=6,
                        output_dtype='bf16', bits=[3, 4], sparkinfer_revision='test',
                        requires_route_preparation=False, paired_boundary='last',
                        descriptor_rows=4, native_info_version=3)
            (directory / 'v41_exl3.json').write_text(json.dumps(meta))
            for name in ('trellis_lut.bin', 'libds41rt_exl3.so'):
                (directory / name).write_bytes(b'fixture')
            variant = {key: meta[key] for key in
                       ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits', 'paired_boundary')}
            variant['directory'] = 'tp4-rank0/m80'
            files = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=package.digest(p))
                     for p in root.rglob('*') if p.is_file()}
            manifest = dict(schema='ds41rt.exl3-package.v1', role='spark', paired_tp4=True,
                            sparkinfer_revision='test', variants=[variant], files=files)
            def write():
                (root / 'manifest.json').write_text(json.dumps(manifest))
            write()
            package.verify(root)
            for field, value in [('paired_tp4', False), ('role', 'coordinator')]:
                original = manifest[field]
                manifest[field] = value
                write()
                with self.assertRaises(ValueError):
                    package.verify(root)
                manifest[field] = original
            variant['paired_boundary'] = 'first'
            write()
            with self.assertRaises(ValueError):
                package.verify(root)

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
