"""Test cached refinement directly from quick calibration, without slow v1 refinement."""
import argparse,hashlib,json,time
from pathlib import Path
from speaking_midi.core import *
from speaking_midi.cached_refine import refine_cached
from experiments.backtest import extra_metrics,fixtures

parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');args=parser.parse_args()
root=Path('outputs/poetry-backtest');piano=Piano()
try:
 for sample in fixtures():
    out=root/sample['name']
    while not (out/'quick.mid').exists():time.sleep(2)
    if (out/'direct.json').exists() and not args.force:continue
    y=load_audio(out/'source.wav');notes=read_midi(out/'quick.mid')
    fitted,trace=refine_cached(piano,y,notes,passes=4,pitch=True)
    audio=render_saved(piano,fitted,out/'cached-direct',len(y)/SR+.5)
    report=dict(extra_metrics(y,audio),notes=len(fitted),trace=trace,
                initial_midi_sha256=hashlib.sha256((out/'quick.mid').read_bytes()).hexdigest(),
                passes=4,refinement='cached-waveform-with-pitch')
    (out/'direct.json').write_text(json.dumps(report,indent=2))
    print('RESULT',sample['name'],{k:v for k,v in report.items() if k!='trace'},flush=True)
finally:piano.close()
