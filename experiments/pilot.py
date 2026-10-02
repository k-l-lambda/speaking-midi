import json,time
from pathlib import Path
from speaking_midi.core import *
from speaking_midi.refine import refine
from speaking_midi.pursuit import fit_log_power
from experiments.backtest import extra_metrics

piano=Piano();atoms,params,config=make_dictionary(piano,'outputs/cache')
for name in ['quiet-night','sonnet-18']:
 out=Path('outputs/pilot')/name;out.mkdir(parents=True,exist_ok=True)
 y=load_audio(Path('examples/poetry')/(name+'.wav'));seconds=len(y)/SR
 notes,trace=fit_log_power(y,atoms,params)
 a=render_saved(piano,notes,out/'power-dictionary',seconds+.5)
 report={'power-dictionary':dict(extra_metrics(y,a),notes=len(notes),trace=trace)}
 notes,trace=refine(piano,y,notes)
 a=render_saved(piano,notes,out/'power-v1',seconds+.5)
 report['power-v1']=dict(extra_metrics(y,a),notes=len(notes),trace=trace)
 (out/'report.json').write_text(json.dumps(report,indent=2));print(name,report,flush=True)
piano.close()
