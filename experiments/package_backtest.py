"""Package all five completed comparisons into a report and an offline listening page."""
import hashlib,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from speaking_midi.core import *
from experiments.backtest import fixtures,extra_metrics

root=Path('outputs/poetry-backtest');rows=[];piano=Piano()
try:
 for sample in fixtures():
    out=root/sample['name'];base=json.loads((out/'report.json').read_text())
    new=json.loads((out/'direct.json').read_text());ablation=json.loads((out/'cached-ablation.json').read_text())
    y=load_audio(out/'source.wav');repeat={}
    for variant in ['v1','cached-direct']:
        notes=read_midi(out/(variant+'.mid'));values=[]
        for _ in range(3):values.append(metrics(y,piano.render(notes,len(y)/SR+.5))['log_band_mae'])
        repeat[variant]=dict(mean=float(np.mean(values)),std=float(np.std(values)),values=values)
    assert base['source_sha256']==hashlib.sha256(Path(sample['path']).read_bytes()).hexdigest()
    rows.append(dict(name=sample['name'],title=sample['title'],language=sample['language'],text=sample['text'],
        seconds=base['seconds'],source_sha256=base['source_sha256'],old=base['results'],new=new,
        ablation=ablation,repeatability=repeat,demo80=json.loads((out/'demo-80.json').read_text())))
finally:piano.close()
asr=json.loads((root/'asr.json').read_text());assert len(asr['results'])==15
pilot={}
for name in ['quiet-night','sonnet-18']:
 pilot[name]=json.loads((Path('outputs/pilot')/name/'report.json').read_text())
summary=dict(protocol=json.loads((root/'protocol.json').read_text()),samples=rows,asr=asr,
             failed_power_pilot=pilot,cpu_check=json.loads((root/'sonnet-18/cpu.json').read_text()))
Path('experiments/poetry-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
labels=['Quiet Night','Spring Dawn','Stork Tower','Sonnet 18','The Tyger'];x=np.arange(5)
fig,axes=plt.subplots(1,2,figsize=(12,4.2))
for ax,key,title in [(axes[0],'log_band_mae','Log-band MAE (optimization target)'),(axes[1],'mr_log_mae','Multi-resolution log-magnitude MAE')]:
 ax.bar(x-.18,[r['old']['v1'][key] for r in rows],.36,label='Original full refinement',color='#6b809b')
 ax.bar(x+.18,[r['new'][key] for r in rows],.36,label='Cached waveform + pitch search',color='#20a580')
 ax.set_xticks(x,labels,rotation=20);ax.set_title(title);ax.set_ylabel('Lower is better');ax.set_ylim(bottom=0);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig('experiments/poetry-comparison.png',dpi=160);plt.close(fig)
parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Poetry fitting backtest</title><style>body{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 16px;background:#101a29;color:#e5eef9}section{padding:18px;margin:20px 0;background:#1b2b41;border-radius:10px}audio{width:100%}.tracks{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}a{color:#82d8bd}summary{cursor:pointer}</style><h1>Poetry fitting backtest</h1><p>Full recordings, identical target calibration and piano preset. Listen before revealing the text. Lower spectral error does not establish intelligible speech.</p>']
for r in rows:
 name=r['name'];old=r['old']['v1'];new=r['new'];gain=100*(1-new['log_band_mae']/old['log_band_mae'])
 parts.append(f'<section><h2>{html.escape(r["title"])}</h2><p>{r["seconds"]:.2f} s · log-band MAE {old["log_band_mae"]:.5f} → {new["log_band_mae"]:.5f} ({gain:.1f}% lower)</p><div class="tracks">')
 for variant,label in [('target','Source (normalized)'),('v1','Original MIDI'),('cached-direct','Improved MIDI')]:
  parts.append(f'<div><h3>{label}</h3><audio controls preload="metadata" src="{name}/{variant}.wav"></audio>')
  if variant!='target':parts.append(f'<a href="{name}/{variant}.mid">Download MIDI</a>')
  parts.append('</div>')
 parts.append(f'</div><details><summary>Show words</summary><p>{html.escape(r["text"])}</p></details></section>')
parts.append('<script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()})));</script>')
(root/'index.html').write_text(''.join(parts))
print('Packaged five full-length comparisons.')
