"""Run dense convolutional fitting on the poetry evaluation fixtures."""
import argparse,json,time
from pathlib import Path
import numpy as np
from speaking_midi.core import Piano, SR, load_audio, render_saved
from speaking_midi.dense import fit_dense
from experiments.backtest import fixtures, extra_metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--only',nargs='+',default=['quiet-night','sonnet-18'])
    ap.add_argument('--nfft',type=int,default=1024)
    ap.add_argument('--device',choices=['cpu','cuda'],default=None)
    ap.add_argument('--cache',type=Path,default=Path('outputs/cache'))
    ap.add_argument('--power',type=int,choices=[1,2],default=2)
    ap.add_argument('--gate',type=float,default=.02)
    ap.add_argument('--step',type=int,default=2)
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--velocity',type=int,default=48)
    ap.add_argument('--floor',type=float,default=.001)
    ap.add_argument('--out',type=Path,default=Path('outputs/dense-floor-high'))
    args=ap.parse_args();piano=Piano()
    try:
        for sample in fixtures():
            if sample['name'] not in args.only:continue
            started=time.monotonic();out=args.out/sample['name'];out.mkdir(parents=True,exist_ok=True)
            y=load_audio(sample['path'])
            notes,report,activation=fit_dense(y,piano,device=args.device,cache=args.cache,nfft=args.nfft,power=args.power,gate=args.gate,
                grid_frames=args.step,steps=args.steps,velocity=args.velocity,floor=args.floor)
            audio=render_saved(piano,notes,out/'dense',len(y)/SR+.5)
            np.savez_compressed(out/'activation.npz',activation=activation)
            report.update(metrics=extra_metrics(y,audio),notes=len(notes),wall_seconds=time.monotonic()-started)
            (out/'report.json').write_text(json.dumps(report,indent=2))
            print('RESULT',sample['name'],report,flush=True)
    finally:piano.close()

if __name__=='__main__':main()
