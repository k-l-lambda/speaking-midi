"""Compare dense and variable-duration MIDI on two local eight-second excerpts.

Only MIDI and numerical evidence are saved; render audio in memory for metrics.
The source files are local inputs, not distributed with the repository.
"""
import json
from pathlib import Path

import numpy as np

from speaking_midi.core import Piano, Note, SR, load_audio, read_midi, write_midi, metrics
from speaking_midi.instrumental import fit_audio


def main():
    output = Path('outputs/instrumental-pilot')
    output.mkdir(parents=True, exist_ok=True)
    piano, results = Piano(), []
    try:
        for name in ['画心', '圣女之歌']:
            full = load_audio(f'outputs/footages/{name}/input.wav')
            start, seconds = 20., 8.
            y = full[round(start*SR):round((start+seconds)*SR)]
            baseline = []
            for n in read_midi(f'footages/{name}.mid'):
                onset, end = max(n.start, start), min(n.start+n.duration, start+seconds)
                if end-onset >= .0005:
                    baseline.append(Note(n.pitch, n.velocity, onset-start, end-onset))
            for label, penalty in [('dense', None), ('sparse003', .03), ('sparse010', .1)]:
                if penalty is None:
                    notes, report = baseline, {}
                else:
                    notes, report = fit_audio(y, piano, device='cuda', steps=600, sparsity=penalty)
                path = output/f'{name}-{label}.mid'
                write_midi(notes, path)
                audio = piano.render(read_midi(path), seconds+.5)
                row = dict(name=name, variant=label, notes=len(notes),
                           mean_duration=float(np.mean([n.duration for n in notes])),
                           metrics=metrics(y, audio), fit=report)
                results.append(row)
                (output/'results.json').write_text(json.dumps(results, indent=2))
                print(json.dumps({k:v for k,v in row.items() if k != 'fit'}, ensure_ascii=False), flush=True)
    finally:
        piano.close()


if __name__ == '__main__':
    main()
