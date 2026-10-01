import tempfile
import unittest
from pathlib import Path
import numpy as np
from talking_midi.core import Piano, Note, write_midi, read_midi, SR, make_dictionary, fit_temporal, metrics

class Roundtrip(unittest.TestCase):
    def test_midi_timing_and_audio(self):
        notes=[Note(60,75,.12,.24),Note(64,45,.36,.12),Note(60,90,.36,.06)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.mid';write_midi(notes,path); decoded=read_midi(path)
            expected=sorted(notes,key=lambda n:(n.start+n.duration,n.pitch))
            for a,b in zip(expected,decoded):
                self.assertEqual(a.pitch,b.pitch); self.assertEqual(a.velocity,b.velocity)
                self.assertAlmostEqual(a.start,b.start,places=5); self.assertAlmostEqual(a.duration,b.duration,places=5)
            piano=Piano(); audio=piano.render(decoded,1)
            self.assertEqual(len(audio),SR); self.assertTrue(np.isfinite(audio).all())
            self.assertGreater(np.max(np.abs(audio)),.001)
            piano.render([Note(90,120,0,.1)],.543)
            repeated=piano.render(decoded,1)
            # FluidSynth has small first-voice/block and int16 dither differences.
            self.assertLess(metrics(audio,repeated)['spectral_convergence'],.01)
            piano.close()

    def test_dictionary_recovers_single_note(self):
        piano=Piano(); atoms,params,_=make_dictionary(piano,'outputs/cache')
        y=piano.render([Note(60,75,0,.12)],.7)
        notes,trace=fit_temporal(y,atoms,params,max_notes=1)
        self.assertEqual(len(notes),1); self.assertEqual(notes[0].pitch,60)
        self.assertEqual(notes[0].velocity,75); self.assertAlmostEqual(notes[0].duration,.12)
        self.assertLessEqual(notes[0].start,.01)
        estimate=piano.render(notes,.7)
        self.assertLess(metrics(y,estimate)['spectral_convergence'],.1)
        piano.close()

if __name__=='__main__': unittest.main()
