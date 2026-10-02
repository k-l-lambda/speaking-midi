"""Reproduce the full Sonnet 18 CPU refinement check from its quick MIDI."""
import json
from pathlib import Path
from speaking_midi.core import Piano, SR, load_audio, read_midi, render_saved
from speaking_midi.cached_refine import refine_cached
from experiments.backtest import extra_metrics

out = Path('outputs/poetry-backtest/sonnet-18')
y = load_audio(out / 'source.wav')
piano = Piano()
try:
    notes, trace = refine_cached(piano, y, read_midi(out / 'quick.mid'),
                                 passes=4, pitch=True, device='cpu')
    audio = render_saved(piano, notes, out / 'cached-cpu', len(y) / SR + .5)
    report = dict(extra_metrics(y, audio), notes=len(notes), trace=trace)
    (out / 'cpu.json').write_text(json.dumps(report, indent=2))
finally:
    piano.close()
