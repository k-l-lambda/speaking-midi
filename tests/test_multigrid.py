import unittest
import numpy as np
import torch
from speaking_midi.core import Note, Piano, SR, metrics
from speaking_midi.multigrid import MultiWindowFeatures, make_banks, segment, REGISTERS


class MultiGridTests(unittest.TestCase):
    def test_analysis_has_local_support(self):
        feature = MultiWindowFeatures('cpu')
        y = np.random.default_rng(0).normal(0, .01, SR).astype(np.float32)
        changed = y.copy()
        changed[SR//2:] *= 100
        a, b = feature(y), feature(changed)
        self.assertEqual(tuple(a.shape), (128, 101))
        torch.testing.assert_close(a[:, :40], b[:, :40], rtol=0, atol=0)

    def test_cpu_recovers_short_high_note_between_coarse_grid_points(self):
        piano = Piano()
        try:
            y = piano.render([Note(96, 64, .13, .02)], .5)
            feature, banks, _ = make_banks(piano, 'cpu', 'outputs/cache', 'single')
            notes, _ = segment(y, feature, banks, 'cpu', steps=200, refit_steps=100)
            self.assertTrue(any(n.pitch == 96 and abs(n.start-.13) < .011
                                and n.duration <= .04+1e-7 for n in notes))
            actual = piano.render(notes, 1.)
            self.assertLess(metrics(y, actual)['log_band_mae'],
                            metrics(y, np.zeros_like(y))['log_band_mae'])
            for n in notes:
                group = next(g for g in REGISTERS if n.pitch in g['pitches'])
                step = group['grid']*.01
                self.assertAlmostEqual(n.start/step, round(n.start/step), places=6)
                self.assertLessEqual(n.duration, max(group['gates'])+1e-7)
        finally:
            piano.close()


if __name__ == '__main__':
    unittest.main()
