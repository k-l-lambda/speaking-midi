"""Experimental cached-waveform refinement; the original CLI remains unchanged."""
import argparse
import json
import time
from pathlib import Path
import soundfile as sf
from .core import Piano,SR,SFONT,load_audio,make_dictionary,fit_temporal,render_saved,metrics
from .refine import refine
from .cached_refine import refine_cached
from .balanced_loss import BalancedLoss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('--out',type=Path,default=Path('outputs/cached-experiment'))
    parser.add_argument('--soundfont',default=SFONT);parser.add_argument('--max-notes',type=int,default=240)
    parser.add_argument('--passes',type=int,default=4);parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--objective',choices=['log','balanced'],default='log')
    parser.add_argument('--balance-weight',type=float,default=.05)
    args=parser.parse_args()
    if not 0 <= args.balance_weight < float('inf'):parser.error('balance-weight must be finite and nonnegative')
    if args.passes<1 or args.max_notes<1:parser.error('passes and max-notes must be positive')
    out=args.out;out.mkdir(parents=True,exist_ok=True);start=time.monotonic();y=load_audio(args.input)
    sf.write(out/'target.wav',y,SR,subtype='FLOAT');piano=Piano(args.soundfont)
    device=None if args.device=='auto' else args.device
    try:
        atoms,params,renderer=make_dictionary(piano,'outputs/cache')
        notes,pursuit=fit_temporal(y,atoms,params,args.max_notes,device=device)
        notes,calibration=refine(piano,y,notes,passes=0)
        quick=render_saved(piano,notes,out/'quick',len(y)/SR+.5)
        objective=(BalancedLoss(y,device=device,spectral_weight=args.balance_weight,
                                envelope_weight=args.balance_weight) if args.objective=='balanced' else None)
        notes,search=refine_cached(piano,y,notes,passes=args.passes,device=device,objective=objective)
        audio=render_saved(piano,notes,out/'refined',len(y)/SR+.5)
        results=dict(quick=metrics(y,quick),refined=metrics(y,audio))
        if objective is not None:
            for name,wave in [('quick',quick),('refined',audio)]:
                results[name].update({k:float(v[0]) for k,v in objective.components([wave]).items()})
                results[name]['combined_objective']=float(objective([wave])[0])
        report=dict(objective=args.objective,balance_weight=args.balance_weight,renderer=renderer,pursuit=pursuit,calibration=calibration,search=search,
                    results=results,
                    notes=len(notes),wall_seconds=time.monotonic()-start)
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report['results'],indent=2))
    finally:piano.close()

if __name__=='__main__':main()
