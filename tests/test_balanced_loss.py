import unittest
import numpy as np
from speaking_midi.balanced_loss import BalancedLoss
from speaking_midi.cached_refine import BatchLogLoss, refine_cached
from speaking_midi.core import Piano, Note, metrics


class BalancedObjective(unittest.TestCase):
    def test_components_and_tail_match_independent_metrics(self):
        rng = np.random.default_rng(12)
        y = rng.normal(0, .02, 24317).astype(np.float32)
        x = y * np.linspace(.2, 1.1, len(y), dtype=np.float32)
        objective = BalancedLoss(y, device='cpu')
        extended = np.r_[x, np.ones(1000, np.float32)]
        result = objective.components([extended])
        expected = metrics(y, extended)
        for key in ['log_band_mae', 'spectral_convergence']:
            self.assertAlmostEqual(float(result[key][0]), expected[key], places=6)
        def envelope(a):
            a = np.pad(a, (0, (-len(a)) % 240))
            return np.sqrt(np.mean(a.reshape(-1, 240)**2, axis=1))
        a, b = envelope(y), envelope(x)
        self.assertAlmostEqual(float(result['envelope_nrmse'][0]),
                               float(np.linalg.norm(a-b)/np.linalg.norm(a)), places=6)
        self.assertAlmostEqual(float(result['envelope_correlation'][0]),
                               float(np.corrcoef(a, b)[0, 1]), places=6)

    def test_zero_weights_and_silent_candidate(self):
        y = np.sin(np.arange(12000)*.1).astype(np.float32)*.03
        candidates = np.stack([y, y*.5, np.zeros_like(y)])
        balanced = BalancedLoss(y, device='cpu', spectral_weight=0, envelope_weight=0)
        np.testing.assert_allclose(balanced(candidates), BatchLogLoss(y, 'cpu')(candidates), atol=1e-6)
        self.assertTrue(np.isfinite(BalancedLoss(y, device='cpu')(candidates)).all())

    def test_full_renderer_improves_custom_objective(self):
        piano = Piano()
        try:
            y = piano.render([Note(60, 75, .02, .12)], .4)
            objective = BalancedLoss(y, device='cpu')
            notes, trace = refine_cached(piano, y, [Note(61, 75, .02, .12)],
                                         passes=1, objective=objective)
            self.assertEqual(notes[0].pitch, 60)
            self.assertLess(trace['verified_loss_history'][-1], trace['verified_loss_history'][0])
            actual = float(objective([piano.render(notes, .9)])[0])
            self.assertAlmostEqual(actual, trace['verified_loss_history'][-1], delta=.002)
        finally:
            piano.close()
