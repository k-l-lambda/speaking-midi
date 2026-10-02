"""Re-render the five existing poetry fixtures with a consistent protocol."""
import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from speaking_midi.core import *
from speaking_midi.refine import refine


def extra_metrics(y, audio):
    estimate=audio[:len(y)]
    # Multi-resolution log-magnitude distances; fixed calibration, no alignment search.
    values=[]
    for nfft in [512,1024,2048]:
        def spec(x):
            return torch.stft(torch.as_tensor(x,dtype=torch.float32),nfft,nfft//4,
                              window=torch.hann_window(nfft),pad_mode='constant',return_complex=True).abs().numpy()
        a=spec(y);b=spec(estimate)
        values.append(float(np.mean(abs(np.log1p(a)-np.log1p(b)))))
    hop=240
    def envelope(x):
        x=np.pad(x,(0,(-len(x))%hop));return np.sqrt(np.mean(x.reshape(-1,hop)**2,axis=1))
    a=envelope(y);b=envelope(estimate)
    correlation=float(np.corrcoef(a,b)[0,1]) if np.std(b)>1e-12 else 0.
    return dict(metrics(y,audio),mr_log_mae=float(np.mean(values)),envelope_correlation=correlation)


def fixtures():
    rows=[]
    folder=Path('examples/poetry')
    for r in json.loads((folder/'prompts.json').read_text()):rows.append(dict(r,path=str(folder/(r['name']+'.wav'))))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs/backtest-v1');ap.add_argument('--only',nargs='*')
    args=ap.parse_args();root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    piano=Piano();atoms,params,config=make_dictionary(piano,'outputs/cache')
    protocol=dict(renderer=config,max_notes=240,torch=torch.__version__,python=platform.python_version(),
                  target_rms=.035,metrics='input interval only; fixed gain; no output normalization or time alignment')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    try:
        for row in fixtures():
            if args.only and row['name'] not in args.only:continue
            out=root/row['name'];out.mkdir(exist_ok=True)
            if (out/'report.json').exists():continue
            shutil.copyfile(row['path'],out/'source.wav');y=load_audio(out/'source.wav');sf.write(out/'target.wav',y,SR,subtype='FLOAT')
            seconds=len(y)/SR;report=dict(sample=row,source_sha256=hashlib.sha256((out/'source.wav').read_bytes()).hexdigest(),seconds=seconds,results={})
            report['results']['silence']=extra_metrics(y,np.zeros_like(y))
            start=time.monotonic();naive=baseline(y);loss=make_log_band_loss(y)
            candidates=[[Note(n.pitch,int(np.clip(n.velocity+d,1,127)),n.start,n.duration) for n in naive] for d in [-36,-24,-12,0,12]]
            naive=min(candidates,key=lambda ns:loss(piano.render(ns,seconds+.5)))
            audio=render_saved(piano,naive,out/'baseline',seconds+.5)
            report['results']['baseline']=dict(extra_metrics(y,audio),notes=len(naive),wall_seconds=time.monotonic()-start)
            start=time.monotonic();notes,trace=fit_temporal(y,atoms,params,240)
            audio=render_saved(piano,notes,out/'dictionary',seconds+.5)
            report['results']['dictionary']=dict(extra_metrics(y,audio),notes=len(notes),wall_seconds=time.monotonic()-start)
            report['pursuit']=trace
            for name,kwargs in [('quick',dict(passes=0)),('v1',dict(passes=1))]:
                fitted,trace=refine(piano,y,notes,**kwargs);audio=render_saved(piano,fitted,out/name,seconds+.5)
                report['results'][name]=dict(extra_metrics(y,audio),notes=len(fitted),wall_seconds=trace['wall_seconds'])
                report[name+'_trace']=trace
            (out/'report.tmp').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            (out/'report.tmp').replace(out/'report.json')
            print('COMPLETED',row['name'],report['results'],flush=True)
    finally:piano.close()

if __name__=='__main__':main()
