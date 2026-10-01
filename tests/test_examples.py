"""Check that the committed demos remain complete and internally consistent."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from speaking_midi.core import read_midi

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and value and '://' not in value:
                self.paths.append(value)


class Examples(unittest.TestCase):
    def test_assets_and_midi(self):
        manifest = json.loads((EXAMPLES / 'manifest.json').read_text())
        self.assertEqual({s['name'] for s in manifest['samples']}, {'en', 'zh'})
        self.assertEqual(manifest['mode'], 'voice_design')
        self.assertEqual(manifest['model'], 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')
        prompts = {row['name']: row for row in json.loads((EXAMPLES / 'prompts.json').read_text())}
        for sample in manifest['samples']:
            name = sample['name']
            self.assertEqual(sample['instruct'], prompts[name]['instruct'])
            self.assertEqual(sample['text'], prompts[name]['text'])
            folder = EXAMPLES / name
            for filename, expected in manifest['assets'][name].items():
                data = (folder / filename).read_bytes()
                self.assertEqual(len(data), expected['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected['sha256'])
            source = sf.info(folder / 'source.wav')
            self.assertEqual(source.samplerate, 24000)
            self.assertEqual(source.channels, 1)
            self.assertGreater(source.duration, 7)
            self.assertLess(source.duration, 9)
            self.assertAlmostEqual(source.duration, sample['seconds'])
            target, sr = sf.read(folder / 'target.wav')
            self.assertEqual(len(target), source.frames)
            self.assertAlmostEqual(float(np.sqrt(np.mean(target ** 2))), .035, places=6)
            metrics = json.loads((folder / 'metrics.json').read_text())
            for variant in ['baseline', 'dictionary', 'refined']:
                audio, rate = sf.read(folder / f'{variant}.wav')
                self.assertEqual(rate, sr)
                self.assertEqual(len(audio), len(target) + sr // 2)
                self.assertTrue(np.isfinite(audio).all())
                self.assertLess(float(np.max(np.abs(audio))), 1)
                notes = read_midi(folder / f'{variant}.mid')
                recorded = json.loads((folder / f'{variant}.json').read_text())
                self.assertEqual(len(notes), metrics['results'][variant]['notes'])
                self.assertEqual(len(notes), len(recorded))
                for note, row in zip(notes, recorded):
                    self.assertEqual(note.pitch, row['pitch'])
                    self.assertEqual(note.velocity, row['velocity'])
                    self.assertAlmostEqual(note.start, row['start'])
                    self.assertAlmostEqual(note.duration, row['duration'])
                    self.assertGreater(note.duration, 0)
                    self.assertLessEqual(note.start + note.duration, source.duration + .0005)

    def test_offline_page_assets_exist(self):
        page = Links()
        page.feed((EXAMPLES / 'index.html').read_text())
        self.assertGreaterEqual(len(page.paths), 10)
        for link in page.paths:
            self.assertTrue((EXAMPLES / link).is_file(), link)


if __name__ == '__main__':
    unittest.main()
