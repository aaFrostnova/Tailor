"""Relocated run root: byte-identical copies, one recorded patch, never redone in place."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main_controls'))
import relocate as RL  # noqa: E402

SCRIPT = ('#!/bin/bash\nCDIR={old}/certify_full; [ x ] && CDIR={old}/certify_full_$DOM\n'
          'python -u {old}/fixed/code/certify_full_frozen.py decode_xenv\n')


class RelocateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.frozen = base / 'frozen'
        for rel in RL.COPY_DIRS:
            (self.frozen / rel).mkdir(parents=True)
        (self.frozen / 'fixed/code/campaign.py').write_text('ROOT = 1\n')
        (self.frozen / 'fixed/code/xenv_static.sh').write_text(SCRIPT.replace('{old}', RL.OLD_PREFIX))
        (self.frozen / 'fixed/code/__pycache__').mkdir()
        (self.frozen / 'fixed/code/__pycache__/x.pyc').write_bytes(b'\0')
        (self.frozen / 'fixed/inputs/C1.json').write_text('{}')
        (self.frozen / 'code/unified_detector.py').write_text('X = 2\n')
        (self.frozen / 'shared_table_20260916/code/shared_greedy.py').write_text('Y = 3\n')
        self.run = base / 'run'

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_unexpected_occurrence_count_is_refused(self):
        text = SCRIPT.replace('{old}', RL.OLD_PREFIX)
        self.assertEqual(RL.patch_text(text, Path('/new'))[1], 3)
        with self.assertRaises(RL.RelocationError):
            RL.patch_text(text + RL.OLD_PREFIX + '\n', Path('/new'))
        with self.assertRaises(RL.RelocationError):
            RL.patch_text(text + str(RL.FROZEN) + '\n', Path('/new'))

    def test_copies_are_identical_and_only_the_script_is_patched(self):
        m = RL.relocate(self.run, frozen=self.frozen)
        paths = sorted(r['path'] for r in m['files'])
        self.assertEqual(paths, ['code/unified_detector.py', 'fixed/code/campaign.py',
                                 'fixed/code/xenv_static.sh', 'fixed/inputs/C1.json',
                                 'shared_table_20260916/code/shared_greedy.py'])
        patched = [r for r in m['files'] if not r['identical']]
        self.assertEqual([r['path'] for r in patched], ['fixed/code/xenv_static.sh'])
        text = (self.run / 'root/fixed/code/xenv_static.sh').read_text()
        self.assertNotIn(RL.OLD_PREFIX, text)
        self.assertIn(str(self.run.resolve() / 'root') + '/fixed/code/certify_full_frozen.py', text)
        for rel in RL.FRESH_DIRS:
            self.assertTrue((self.run / 'root' / rel).is_dir())
        self.assertFalse((self.run / 'root/fixed/code/__pycache__').exists())
        RL.verify(self.run, frozen=self.frozen)
        with self.assertRaises(RL.RelocationError):
            RL.relocate(self.run, frozen=self.frozen)

    def test_verify_detects_a_changed_copy_and_a_changed_source(self):
        RL.relocate(self.run, frozen=self.frozen)
        (self.run / 'root/code/unified_detector.py').write_text('X = 3\n')
        with self.assertRaises(RL.RelocationError):
            RL.verify(self.run, frozen=self.frozen)
        (self.run / 'root/code/unified_detector.py').write_text('X = 2\n')
        RL.verify(self.run, frozen=self.frozen)
        (self.frozen / 'fixed/inputs/C1.json').write_text('{"a": 1}')
        with self.assertRaises(RL.RelocationError):
            RL.verify(self.run, frozen=self.frozen)

    def test_relocation_record_is_json(self):
        RL.relocate(self.run, frozen=self.frozen)
        rec = json.loads((self.run / 'relocation.json').read_text())
        self.assertEqual(rec['frozen_root'], str(self.frozen))


if __name__ == '__main__':
    unittest.main()
