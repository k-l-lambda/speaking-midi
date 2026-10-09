"""Generate full listening candidates; validate saved MIDI and render in memory."""
import json
from pathlib import Path
import mido
import numpy as np
from speaking_midi.core import Piano, SR, load_audio, read_midi
from speaking_midi.instrumental import save_midi
from speaking_midi.multigrid import fit_multigrid
from experiments.chime_objective import SOURCE, measure


def main():
    out = Path('outputs/multigrid')
    cases = [('半岛铁盒-分音区时间粒度', SOURCE/'半岛铁盒.mp3', None),
             ('画心-前38秒-分音区时间粒度', Path('outputs/footages/画心/input.wav'), 38)]
    results = []
    for name, source, seconds in cases:
        path = out/f'{name}.mid'
        if path.exists():
            raise FileExistsError(path)
        y = load_audio(source, target_rms=None)
        if seconds is not None:
            y = y[:round(seconds*SR)]
        piano = Piano()
        try:
            notes, report = fit_multigrid(y, piano, device='cuda', steps=600, analysis='single')
            save_midi(notes, path, len(y)/SR)
            saved = read_midi(path)
            assert all(n.duration > 0 and 21 <= n.pitch <= 108 for n in saved)
            assert abs(mido.MidiFile(path).length-len(y)/SR) <= .001
            audio = piano.render(saved, len(y)/SR+.5)
            assert np.isfinite(audio).all() and np.max(np.abs(audio)) < .999
            row = dict(file=str(path), source=str(source), seconds=len(y)/SR,
                       normalization='fixed', gain=1, **measure(y, audio, saved))
            results.append(row)
            report['validation'] = row
            path.with_suffix('.report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
            (out/'full-results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            piano.close()


if __name__ == '__main__':
    main()
