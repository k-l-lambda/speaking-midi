"""Verify selected dense MIDI through two fresh full-renderer/ASR repetitions."""
from pathlib import Path
from speaking_midi.core import Piano, SR, load_audio, read_midi, render_saved
from experiments.backtest import fixtures

for repeat in [1,2]:
    piano=Piano()
    try:
        for sample in fixtures():
            root=Path('outputs/dense-repeat-'+str(repeat))/sample['name']
            root.mkdir(parents=True,exist_ok=True)
            notes=read_midi(Path('outputs/dense-floor-high')/sample['name']/'dense.mid')
            render_saved(piano,notes,root/'dense',len(load_audio(sample['path']))/SR+.5)
    finally:piano.close()
