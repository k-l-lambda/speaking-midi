"""Measure the current demo's default 80-iteration, global-refinement configuration."""
import json,time
from pathlib import Path
from speaking_midi.core import *
from speaking_midi.refine import refine
from experiments.backtest import fixtures,extra_metrics
root=Path('outputs/poetry-backtest');p=Piano();atoms,params,_=make_dictionary(p,'outputs/cache')
try:
 for row in fixtures():
    out=root/row['name'];y=load_audio(out/'source.wav');start=time.monotonic()
    notes,trace=fit_temporal(y,atoms,params,80);notes,calibration=refine(p,y,notes,passes=0)
    a=render_saved(p,notes,out/'demo-80',len(y)/SR+.5)
    report=dict(extra_metrics(y,a),notes=len(notes),wall_seconds=time.monotonic()-start,
                pursuit=trace,calibration=calibration)
    (out/'demo-80.json').write_text(json.dumps(report,indent=2));print(row['name'],report['log_band_mae'],flush=True)
finally:p.close()
