import unittest
import numpy as np
import torch
from speaking_midi.instrumental_loss import InstrumentalLoss
from speaking_midi.instrumental import fit_audio
from speaking_midi.core import Piano,Note,SR,metrics


class ConservativeLossTests(unittest.TestCase):
    def test_distant_frame_does_not_change_local_scale_or_gradient(self):
        target=torch.full((1,128,20),.1)
        changed=target.clone();changed[:,:,15:]*=100
        estimates=[torch.full_like(target,.2,requires_grad=True) for _ in range(2)]
        losses=[InstrumentalLoss(t,[(60,.24)],objective='conservative',treble_cost=.1)
                for t in [target,changed]]
        for loss,estimate in zip(losses,estimates):
            terms=loss(estimate,torch.ones(1,1,10),.03)
            sum(terms).backward()
        torch.testing.assert_close(losses[0].floor[:,:,:10],losses[1].floor[:,:,:10],rtol=0,atol=0)
        torch.testing.assert_close(losses[0].scale[:,:,:10],losses[1].scale[:,:,:10],rtol=0,atol=0)
        torch.testing.assert_close(estimates[0].grad[:,:,:10],estimates[1].grad[:,:,:10],rtol=0,atol=0)

    def test_overshoot_is_more_expensive_than_equal_log_undershoot(self):
        target=torch.ones(1,128,10)
        loss=InstrumentalLoss(target,[(60,.24)],objective='conservative')
        target_log=torch.log1p(target/loss.floor)
        high=loss.floor*torch.expm1(target_log+.3)
        low=loss.floor*torch.expm1(target_log-.3)
        activation=torch.zeros(1,1,5)
        self.assertGreater(float(loss(high,activation,0)[0]),float(loss(low,activation,0)[0]))

    def test_soft_pitch_cost_and_cpu_note_recovery(self):
        target=torch.ones(1,128,10)
        loss=InstrumentalLoss(target,[(60,.24),(96,.24)],objective='conservative',treble_cost=.1)
        lo=torch.tensor([[[1.],[0.]]]);hi=torch.tensor([[[0.],[1.]]])
        self.assertGreater(float(loss(target,hi,.03)[1]),float(loss(target,lo,.03)[1]))
        p=Piano()
        try:
            y=p.render([Note(60,48,.12,.24)],.8)
            notes,report=fit_audio(y,p,device='cpu',steps=200,objective='conservative',treble_cost=.1)
            self.assertTrue(any(n.pitch==60 and abs(n.start-.12)<.041 for n in notes))
            actual=p.render(notes,1.3)
            self.assertLess(metrics(y,actual)['log_band_mae'],metrics(y,np.zeros_like(y))['log_band_mae'])
            self.assertEqual(report['objective'],'conservative')
        finally:p.close()

if __name__=='__main__':unittest.main()
