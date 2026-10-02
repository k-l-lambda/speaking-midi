import tempfile
import unittest
from pathlib import Path
import numpy as np
from speaking_midi.core import Piano, Note, SR, write_midi, read_midi, metrics
from speaking_midi.dense import decode_activations, fit_dense


class DenseFitting(unittest.TestCase):
    def test_calibration_and_nonoverlapping_midi_grid(self):
        coefficients=np.array([[0.,.3,.6,.15],[.9,0.,0.,0.]])
        calibration=np.array([[0.,.1,.2,.4,1.],[0.,.2,.4,.8,2.]])
        notes=decode_activations(coefficients,calibration,[60,64],[1,4,8,16,32],.02,2,.08)
        self.assertEqual([(n.pitch,n.velocity) for n in notes],[(60,12),(60,21),(64,17)])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'control.mid';write_midi(notes,path)
            decoded=read_midi(path)
            self.assertEqual(len(decoded),3)
            for n in decoded:
                self.assertGreaterEqual(n.start,0)
                self.assertLessEqual(n.start+n.duration,.08+1e-9)

    def test_single_piano_note_recovery_on_cpu(self):
        piano=Piano()
        try:
            y=piano.render([Note(60,64,.06,.02)],.3)
            notes,report,_=fit_dense(y,piano,device='cpu',steps=150)
            self.assertEqual(report['optimization']['device'],'cpu')
            self.assertLess(report['loss_history'][-1][1],report['loss_history'][0][1])
            self.assertTrue(any(n.pitch==60 and abs(n.start-.06)<=.02 for n in notes))
            rendered=piano.render(notes,.8)
            self.assertLess(metrics(y,rendered)['log_band_mae'],metrics(y,np.zeros_like(y))['log_band_mae'])
        finally:piano.close()

    def test_reject_overlapping_gate(self):
        with self.assertRaises(ValueError):
            fit_dense(np.ones(SR,dtype=np.float32),None,gate=.04,grid_frames=2)

if __name__=='__main__':unittest.main()
