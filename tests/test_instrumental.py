import tempfile
import unittest
from pathlib import Path

import numpy as np
import mido

from speaking_midi.core import Note, Piano, SR, read_midi, write_midi, metrics
from speaking_midi.instrumental import decode, fit_audio, save_midi


class InstrumentalFitting(unittest.TestCase):
    def test_conflicting_durations_and_midi_release(self):
        values = np.zeros((2, 10))
        values[0, 0] = .5
        values[1, 0] = .8
        values[0, 2] = .5
        values[0, 7] = .5
        levels = [1, 4, 8, 12, 48]
        calibration = np.tile([0, .01, .02, .03, 1], (2, 1))
        notes = decode(values, calibration, [(60, .08), (60, .24)], levels, .32)
        self.assertEqual(len(notes), 2)
        self.assertAlmostEqual(notes[0].duration, .24)
        self.assertAlmostEqual(notes[1].duration, .04)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'notes.mid'
            write_midi(notes, path)
            restored = read_midi(path)
            self.assertEqual(len(restored), 2)
            self.assertAlmostEqual(restored[0].duration, .24, places=3)
            self.assertAlmostEqual(restored[1].start+restored[1].duration, .32, places=3)
            save_midi(notes, path, .8)
            self.assertAlmostEqual(mido.MidiFile(path).length, .8, places=3)
            self.assertEqual(len(read_midi(path)), 2)

    def test_sustained_note_cpu_recovery(self):
        piano = Piano()
        try:
            y = piano.render([Note(60, 48, .12, .24)], .8)
            notes, report = fit_audio(y, piano, device='cpu', steps=200, sparsity=.03)
            self.assertTrue(any(n.pitch == 60 and abs(n.start-.12) <= .04
                                and n.duration >= .24 for n in notes))
            rendered = piano.render(notes, 1.3)
            self.assertLess(metrics(y, rendered)['log_band_mae'],
                            metrics(y, np.zeros_like(y))['log_band_mae'])
            self.assertLess(report['chunks'][0]['history'][-1]['spectral'],
                            report['chunks'][0]['history'][0]['spectral'])
        finally:
            piano.close()


if __name__ == '__main__':
    unittest.main()
