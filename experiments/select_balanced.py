"""Collect the frozen pilot choice for listening and ASR, without re-optimizing."""
import hashlib
import json
import shutil
from pathlib import Path
from experiments.backtest import fixtures

selection = json.loads(Path('experiments/balanced-selection.json').read_text())
weight = selection['selected_weight']
for sample in fixtures():
    name = sample['name']
    source = Path('outputs/balanced-search')/name/f'w{weight:g}'
    out = Path('outputs/balanced-selected')/name
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((source/'report.json').read_text())
    quick = Path('outputs/poetry-backtest')/name/'quick.mid'
    assert report['initial_midi_sha256'] == hashlib.sha256(quick.read_bytes()).hexdigest()
    assert report['weight'] == weight
    for suffix in ['wav', 'mid', 'json']:
        shutil.copyfile(source/f'balanced.{suffix}', out/f'balanced.{suffix}')
    shutil.copyfile(source/'report.json', out/'report.json')
print('Collected five outputs with the frozen weight:', weight)
