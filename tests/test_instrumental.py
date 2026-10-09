import tempfile
import unittest
from pathlib import Path

import numpy as np
import mido
import soundfile as sf

from speaking_midi.core import Note, Piano, SR, read_midi, write_midi, metrics, load_audio
from speaking_midi.instrumental import decode, fit_audio, save_midi


class InstrumentalFitting(unittest.TestCase):
    def test_fixed_gain_is_independent_of_distant_audio_and_file_length(self):
        wave = .02*np.sin(2*np.pi*220*np.arange(4*SR)/SR)
        changed = wave.copy()
        changed[3*SR:] *= 30
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'{name}.wav' for name in ['original', 'loud-tail', 'prefix']]
            for path, samples in zip(paths, [wave, changed, wave[:2*SR]]):
                sf.write(path, samples, SR, subtype='FLOAT')
            fixed = [load_audio(path, target_rms=None, gain=.5) for path in paths]
            np.testing.assert_array_equal(fixed[0][:2*SR], fixed[1][:2*SR])
            np.testing.assert_array_equal(fixed[0][:2*SR], fixed[2])
            np.testing.assert_allclose(fixed[2],wave[:2*SR]*.5,atol=1e-8)
            # This input would have reproduced the old distant-signal coupling.
            old = [load_audio(path) for path in paths]
            self.assertGreater(np.linalg.norm(old[0][:2*SR]-old[1][:2*SR]),1)

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
