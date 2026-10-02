import json,time
from pathlib import Path
from speaking_midi.core import *
from speaking_midi.cached_refine import refine_cached
from experiments.backtest import extra_metrics,fixtures

root=Path('outputs/poetry-backtest');piano=Piano()
try:
 for sample in fixtures():
    out=root/sample['name']
    while not (out/'report.json').exists():time.sleep(2)
    if (out/'cached-ablation.json').exists():continue
    y=load_audio(out/'source.wav');seconds=len(y)/SR;original=read_midi(out/'v1.mid')
    report={}
    for name,pitch in [('cached-no-pitch',False),('cached-pitch',True)]:
        notes,trace=refine_cached(piano,y,original,passes=2,pitch=pitch)
        audio=render_saved(piano,notes,out/name,seconds+.5)
        report[name]=dict(extra_metrics(y,audio),notes=len(notes),trace=trace)
        print('RESULT',sample['name'],name,{k:v for k,v in report[name].items() if k!='trace'},flush=True)
    (out/'cached-ablation.json').write_text(json.dumps(report,indent=2))
finally:piano.close()
