"""Held-out image freezing: disjoint by path, file hash and pixel hash; deterministic; refuses shortfalls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'heldout'))
import freeze_images as F  # noqa: E402


class FreezeImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.pool = base / 'pool'
        for s in ('A', 'B'):
            (self.pool / s / 'img').mkdir(parents=True)
            for i in range(6):
                Image.new('RGB', (64, 64), (10 * i + (s == 'B'), 0, 0)).save(self.pool / s / 'img' / f'{s}_{i:05d}.png')
        # B_00005 shows the same picture as A_00000 under another name and with other bytes
        Image.new('RGB', (64, 64), (0, 0, 0)).save(self.pool / 'B' / 'img' / 'B_00005.png', compress_level=1)
        a0 = self.pool / 'A' / 'img' / 'A_00000.png'
        self.manifest = base / 'm.json'
        self.manifest.write_text(json.dumps(dict(splits=dict(selection=dict(images=[dict(
            path='/home/x/pool/A/img/A_00000.png', sha256=F.file_sha256(a0), pixel_sha256=F.pixel_sha256(a0))])))))
        self.p = [patch.object(F, 'POOL', self.pool), patch.object(F, 'BASE_MANIFESTS', (self.manifest,))]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()
        self.tmp.cleanup()

    def test_excludes_by_path_and_by_pixel_duplicate(self):
        doc = F.freeze(Path(self.tmp.name) / 'out.json', 'heldout', {'A': 5, 'B': 5}, 7)
        names = [r['name'] for r in doc['images']]
        self.assertNotIn('A/img/A_00000.png', names)
        self.assertNotIn('B/img/B_00005.png', names)
        self.assertEqual(len(names), 10)
        self.assertEqual([r['index'] for r in doc['images']], list(range(10)))

    def test_deterministic_and_frozen_once(self):
        a = F.freeze(Path(self.tmp.name) / 'a.json', 'heldout', {'A': 3, 'B': 3}, 7)
        b = F.freeze(Path(self.tmp.name) / 'b.json', 'heldout', {'A': 3, 'B': 3}, 7)
        self.assertEqual(a['images_sha256'], b['images_sha256'])
        with self.assertRaises(SystemExit):
            F.freeze(Path(self.tmp.name) / 'a.json', 'heldout', {'A': 3, 'B': 3}, 7)

    def test_a_later_set_excludes_an_earlier_one_and_shortfalls_are_refused(self):
        first = Path(self.tmp.name) / 'first.json'
        F.freeze(first, 'heldout', {'A': 3, 'B': 3}, 7)
        second = F.freeze(Path(self.tmp.name) / 'second.json', 'negatives', {'A': 2, 'B': 2}, 7, [first])
        taken = {r['pixel_sha256'] for r in json.loads(first.read_text())['images']}
        self.assertFalse(taken & {r['pixel_sha256'] for r in second['images']})
        with self.assertRaises(SystemExit):
            F.freeze(Path(self.tmp.name) / 'third.json', 'more', {'A': 5}, 7, [first])

    def test_a_changed_pool_image_is_refused(self):
        Image.new('RGB', (64, 64), (1, 2, 3)).save(self.pool / 'A' / 'img' / 'A_00000.png')
        with self.assertRaises(SystemExit):
            F.freeze(Path(self.tmp.name) / 'out.json', 'heldout', {'A': 2}, 7)


if __name__ == '__main__':
    unittest.main()
