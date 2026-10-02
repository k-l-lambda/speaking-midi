"""Package the balanced-objective follow-up, including all exploratory trials."""
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
from speaking_midi.core import Piano, SR, load_audio, read_midi
from speaking_midi.balanced_loss import BalancedLoss

root = Path('outputs/balanced-selected')
old = json.loads(Path('experiments/poetry-results.json').read_text())
selection = json.loads(Path('experiments/balanced-selection.json').read_text())
asr = json.loads((root/'asr.json').read_text())
assert len(asr['results']) == 5
rows = []
piano = Piano()
try:
    for previous in old['samples']:
        name = previous['name']
        out = root/name
        y = load_audio(Path('examples/poetry')/(name+'.wav'))
        objective = BalancedLoss(y, device='cpu', spectral_weight=.05, envelope_weight=.05)
        variants = {}
        for label, path in [('original', Path('outputs/poetry-backtest')/name/'v1'),
                            ('log_only', Path('outputs/poetry-backtest')/name/'cached-direct'),
                            ('balanced', out/'balanced')]:
            audio, sr = sf.read(path.with_suffix('.wav'), dtype='float32')
            assert sr == SR
            variants[label] = {k: float(v[0]) for k, v in objective.components([audio]).items()}
        repeats = []
        notes = read_midi(out/'balanced.mid')
        for _ in range(3):
            audio = piano.render(notes, len(y)/SR+.5)
            repeats.append({k: float(v[0]) for k, v in objective.components([audio]).items()})
        result = json.loads((out/'report.json').read_text())
        asr_row = next(x for x in asr['results'] if x['sample'] == name)
        assert asr_row['audio_sha256'] == hashlib.sha256((out/'balanced.wav').read_bytes()).hexdigest()
        rows.append(dict(name=name, title=previous['title'], seconds=previous['seconds'],
                         text=previous['text'], old=previous['old']['v1'], log_only=previous['new'],
                         balanced=result, rescored=variants, repeated_renders=repeats,
                         asr=asr_row, source_sha256=previous['source_sha256']))
finally:
    piano.close()
pilots = {name: {p.parent.name: json.loads(p.read_text()) for p in
                 sorted((Path('outputs/balanced-search')/name).glob('*/report.json'))}
          for name in selection['pilot_samples']}
ablation = {component: {name: json.loads((Path('outputs/balanced-ablation')/component/name/'w0.05/report.json').read_text())
                       for name in selection['pilot_samples']} for component in ['spectral', 'envelope']}
report = dict(selection=selection, samples=rows, pilots=pilots, ablation=ablation, asr_protocol=asr,
              cpu=json.loads(Path('outputs/balanced-cpu/sonnet-18/w0.05/report.json').read_text()))
Path('experiments/balanced-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=False)
for ax, row in zip(axes, rows):
    name = row['name']
    for label, path, color in [
        ('Source', Path('outputs/poetry-backtest')/name/'target.wav', '#506078'),
        ('Log-only cached', Path('outputs/poetry-backtest')/name/'cached-direct.wav', '#d88932'),
        ('Balanced cached', root/name/'balanced.wav', '#189d7c')]:
        audio, sr = sf.read(path, dtype='float32')
        audio = audio[:round(row['seconds']*sr)]
        audio = np.pad(audio, (0, (-len(audio)) % 240))
        envelope = np.sqrt(np.mean(audio.reshape(-1, 240)**2, axis=1))
        ax.plot(np.arange(len(envelope))*.01, envelope, label=label, color=color, lw=1.2)
    ax.set_title(name);ax.set_ylabel('10 ms RMS');ax.grid(alpha=.2)
axes[0].legend(ncol=3);axes[-1].set_xlabel('Seconds')
fig.tight_layout();fig.savefig('experiments/balanced-envelopes.png', dpi=130);plt.close(fig)
parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Balanced piano fitting</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 16px;background:#101a29;color:#e5eef9}section{padding:20px;margin:20px 0;background:#1b2b41;border-radius:10px}.tracks{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}audio{width:100%}a{color:#82d8bd}</style><h1>Balanced piano fitting</h1><p>Full recordings, fixed playback gain. Listen before revealing the words. Acoustic improvements do not establish intelligibility.</p>']
for row in rows:
    name = row['name']
    parts.append(f'<section><h2>{html.escape(row["title"])}</h2><div class="tracks">')
    for label, stem in [('Source', f'../poetry-backtest/{name}/target'),
                        ('Original', f'../poetry-backtest/{name}/v1'),
                        ('Log-only cached', f'../poetry-backtest/{name}/cached-direct'),
                        ('Balanced cached', f'{name}/balanced')]:
        parts.append(f'<div><h3>{label}</h3><audio controls preload="metadata" src="{stem}.wav"></audio>')
        if label != 'Source':parts.append(f'<a href="{stem}.mid">MIDI</a>')
        parts.append('</div>')
    parts.append(f'</div><details><summary>Show words</summary><p>{html.escape(row["text"])}</p></details></section>')
parts.append('<script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()})));</script>')
(root/'index.html').write_text(''.join(parts))
print('Packaged balanced objective results for all five recordings.')
