import unittest
import numpy as np
from speaking_midi.core import Piano,Note,SR,metrics,make_log_band_loss
from speaking_midi.cached_refine import NoteCache,BatchLogLoss,refine_cached

class CachedRefinement(unittest.TestCase):
    def test_block_phase_additivity(self):
        p=Piano()
        try:
            notes=[Note(60,75,.013,.12),Note(64,65,.079,.24),Note(67,45,.12,.1)]
            actual=p.render(notes,.7);approx=NoteCache(p).render(notes,.7)
            self.assertLess(metrics(actual,approx)['spectral_convergence'],.02)
        finally:p.close()

    def test_batched_loss_matches_original(self):
        rng=np.random.default_rng(123);y=rng.normal(0,.02,SR//2).astype(np.float32)
        candidates=np.stack([y,np.zeros_like(y),y*.7])
        batch=BatchLogLoss(y,device='cpu')(candidates)
        scalar=make_log_band_loss(y)
        np.testing.assert_allclose(batch,[scalar(a) for a in candidates],rtol=1e-5,atol=1e-6)

    def test_pitch_repair_is_verified_by_real_renderer(self):
        p=Piano()
        try:
            y=p.render([Note(60,75,.02,.12)],.4)
            before=[Note(61,75,.02,.12)]
            notes,trace=refine_cached(p,y,before,passes=1,pitch=True,device='cpu')
            self.assertEqual(notes[0].pitch,60)
            self.assertLess(trace['verified_loss_history'][-1],trace['verified_loss_history'][0])
            self.assertTrue(trace['passes'][0]['accepted'])
        finally:p.close()

if __name__=='__main__':unittest.main()
