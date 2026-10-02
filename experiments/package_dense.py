"""Audit dense MIDI/ASR evidence and package portable listening artifacts."""
import hashlib, html, json, shutil
from pathlib import Path
import mido
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.backtest import fixtures
from experiments.asr_scoring import score


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def select(report, name, variant):
    return next(x for x in report['results'] if x['sample']==name and x['variant']==variant)

def pooled(records, language):
    chosen=[x for x in records if x['language']==language]
    errors=sum(x['errors'] for x in chosen);units=sum(x['reference_units'] for x in chosen)
    return dict(errors=errors,reference_units=units,error_rate=errors/units)

original=read('experiments/poetry-results.json')
balanced=read('experiments/balanced-results.json')
root=Path('outputs/dense-floor-high')
asr=read(root/'asr.json');repeats=[read(f'outputs/dense-repeat-{i}/asr.json') for i in [1,2]]
fixed=read(root/'asr-temperature0.json');fixed_old=read('outputs/poetry-backtest/asr-temperature0.json')
selection=read('experiments/dense-selection.json')
cpu_asr=read('outputs/dense-cold-cpu/asr.json')
assert len(cpu_asr['results'])==5
assert len(asr['results'])==5 and all(len(x['results'])==5 for x in repeats)
assert len(fixed['results'])==5 and len(fixed_old['results'])==10
assets=Path('examples/poetry/dense');assets.mkdir(parents=True,exist_ok=True)
rows=[];manifest=[]
for sample in fixtures():
    name=sample['name'];folder=root/name;previous=next(x for x in original['samples'] if x['name']==name)
    result=read(folder/'report.json')
    for key in ['power','gate','nfft','velocity']:
        assert result['config'][key]==selection['config'][key]
    assert result['optimization']['floor']==selection['config']['floor']
    assert result['optimization']['grid_frames']==selection['config']['step']
    assert sha(sample['path'])==previous['source_sha256']
    messages=list(mido.MidiFile(folder/'dense.mid'))
    assert all(m.type in ['set_tempo','program_change','note_on','note_off','end_of_track'] for m in messages)
    assert all(m.program==0 and m.channel==0 for m in messages if m.type=='program_change')
    assert all(m.channel==0 and 21<=m.note<=108 for m in messages if m.type in ['note_on','note_off'])
    assert sum(m.type=='note_on' and m.velocity>0 for m in messages)==result['notes']
    info=sf.info(folder/'dense.wav')
    assert info.samplerate==24000 and abs(info.duration-previous['seconds']-.5)<1/24000
    evaluations={
        'source':select(original['asr'],name,'source'),
        'original':select(original['asr'],name,'v1'),
        'balanced':next(x['asr'] for x in balanced['samples'] if x['name']==name),
        'dense':select(asr,name,'dense'),
        'cpu':select(cpu_asr,name,'dense'),
        'repeat1':select(repeats[0],name,'dense'),
        'repeat2':select(repeats[1],name,'dense'),
        'fixed_original':select(fixed_old,name,'v1'),
        'fixed_source':select(fixed_old,name,'source'),
        'fixed_dense':select(fixed,name,'dense')}
    for label,item in evaluations.items():
        computed=score(sample['text'],item['hypothesis'],sample['language'])
        assert abs(computed['error_rate']-item['error_rate'])<1e-10
        evaluations[label]=dict(item,**{k:v for k,v in computed.items() if k!='error_rate'},language=sample['language'])
    for label,path in [('dense',folder/'dense.wav'),('fixed_dense',folder/'dense.wav'),
                       ('cpu',Path('outputs/dense-cold-cpu')/name/'dense.wav'),
                       ('repeat1',Path('outputs/dense-repeat-1')/name/'dense.wav'),
                       ('repeat2',Path('outputs/dense-repeat-2')/name/'dense.wav')]:
        assert evaluations[label]['audio_sha256']==sha(path)
    for suffix in ['mid','wav']:
        shutil.copyfile(folder/f'dense.{suffix}',assets/f'{name}.{suffix}')
    manifest.append(dict(name=name,source_sha256=sha(sample['path']),midi_sha256=sha(assets/f'{name}.mid'),
                         audio_sha256=sha(assets/f'{name}.wav'),notes=result['notes'],seconds=info.duration))
    rows.append(dict(name=name,title=sample['title'],language=sample['language'],text=sample['text'],
                     source_seconds=previous['seconds'],original=previous['old']['v1'],
                     dense=result,asr=evaluations))
labels=list(rows[0]['asr'])
aggregate={label:{language:pooled([x['asr'][label] for x in rows],language)
                  for language in ['Chinese','English']} for label in labels}
checks={}
for method in ['dense','repeat1','repeat2','fixed_dense','cpu']:
    baseline='fixed_original' if method=='fixed_dense' else 'original'
    checks[method]={language:aggregate[baseline][language]['error_rate']-aggregate[method][language]['error_rate']>=.2
                    for language in ['Chinese','English']}
assert all(all(v.values()) for v in checks.values()), 'Material improvement has not been established in every audit.'
pilots={name:dict(asr=read(Path('outputs')/name/'asr.json'),
                 results={sample:read(Path('outputs')/name/sample/'report.json') for sample in selection['pilot_samples']})
        for name in selection['pilot_runs']}
summary=dict(selection=selection,samples=rows,aggregate=aggregate,material_improvement_checks=checks,
             pilots=pilots,asr_protocol=asr,fixed_protocol=fixed,
             cpu=dict(warm=read('outputs/dense-cpu-eval/sonnet-18/report.json'),
                      cold=read('outputs/dense-cold-cpu/sonnet-18/report.json'),
                      warm_asr=read('outputs/dense-cpu-eval/asr.json'),cold_asr=cpu_asr,
                      all_results={x['name']:read(Path('outputs/dense-cold-cpu')/x['name']/'report.json') for x in rows}),
             cli_matches_selected_midi=sha('outputs/dense-cli-check/dense.mid')==sha(root/'sonnet-18/dense.mid'))
assert summary['cli_matches_selected_midi']
assert summary['cpu']['cold']['dictionary']['cache_hit'] is False
Path('experiments/dense-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(assets/'manifest.json').write_text(json.dumps(dict(selection=selection,files=manifest),ensure_ascii=False,indent=2))
x=np.arange(5);fig,ax=plt.subplots(figsize=(11,4.5))
for offset,label,color in [(-.25,'original','#6b809b'),(0,'balanced','#d88932'),(.25,'dense','#189d7c')]:
    ax.bar(x+offset,[100*r['asr'][label]['error_rate'] for r in rows],.24,label=label,color=color)
ax.set_xticks(x,['Quiet Night','Spring Dawn','Stork Tower','Sonnet 18','The Tyger'])
ax.set_ylabel('Raw CER / WER (%) — lower is better');ax.set_ylim(0,145);ax.grid(axis='y',alpha=.2)
ax.legend();fig.tight_layout();fig.savefig('experiments/dense-asr.png',dpi=150);plt.close(fig)
parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Speaking MIDI: dense speech fitting</title><style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 16px;background:#101a29;color:#e5eef9}section{padding:20px;margin:20px 0;background:#1b2b41;border-radius:10px}.tracks{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px}audio{width:100%}a{color:#82d8bd}</style><h1>Speaking MIDI: dense speech fitting</h1><p>Fixed Grand Piano MIDI only. No source speech layer, pitch bend, or post-processing. Listen before revealing the words; ASR improvement does not imply perfect intelligibility.</p>']
for row in rows:
    name=row['name'];parts.append(f'<section><h2>{html.escape(row["title"])}</h2><div class="tracks">')
    for label,path in [('Source speech',f'../{name}.wav'),('Dense piano MIDI',f'{name}.wav')]:
        parts.append(f'<div><h3>{label}</h3><audio controls preload="metadata" src="{path}"></audio></div>')
    parts.append(f'</div><p><a href="{name}.mid">Download MIDI</a> · {row["dense"]["notes"]} notes</p><details><summary>Show reference and ASR</summary><p>Reference: {html.escape(row["text"])}</p><p>ASR: {html.escape(row["asr"]["dense"]["hypothesis"])}</p></details></section>')
parts.append('<script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()})));</script>')
(assets/'index.html').write_text(''.join(parts))
print(json.dumps(aggregate,indent=2));print('All material-improvement checks passed.')
