import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import soundfile as sf
from .core import *

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('--out',default='outputs/experiment')
    ap.add_argument('--soundfont',default=SFONT); ap.add_argument('--max-notes',type=int,default=240)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    started=time.time(); y=load_audio(args.input); seconds=len(y)/SR
    sf.write(out/'target.wav',y,SR,subtype='FLOAT')
    piano=Piano(args.soundfont)
    atoms,params,config=make_dictionary(piano,'outputs/cache')
    print('dictionary',atoms.shape,flush=True)
    naive=baseline(y)
    baseline_trials=[]; best_naive=naive; naive_loss=float('inf')
    for dv in [-36,-24,-12,0,12]:
        candidate=[replace(n,velocity=int(np.clip(n.velocity+dv,1,127))) for n in naive]
        m=metrics(y,piano.render(candidate,seconds+.5))
        baseline_trials.append(dict(velocity_delta=dv,**m))
        if m['log_band_mae']<naive_loss: best_naive=candidate; naive_loss=m['log_band_mae']
    naive=best_naive
    audio=render_saved(piano,naive,out/'baseline',seconds+.5)
    results={'baseline':dict(metrics(y,audio),notes=len(naive))}
    fitted,trace=fit_temporal(y,atoms,params,args.max_notes)
    audio=render_saved(piano,fitted,out/'dictionary',seconds+.5)
    results['dictionary']=dict(metrics(y,audio),notes=len(fitted))
    # Exact renderer-in-the-loop discrete velocity/duration coordinate sweeps.
    # A candidate is accepted only if the measured log-band loss decreases.
    best=fitted; best_loss=results['dictionary']['log_band_mae']; trials=[]
    for dv in [-12,-6,6,12]:
        candidate=[replace(n,velocity=int(np.clip(n.velocity+dv,1,127))) for n in fitted]
        a=piano.render(candidate,seconds+.5); m=metrics(y,a)
        trials.append(dict(velocity_delta=dv,**m))
        if m['log_band_mae']<best_loss: best=candidate; best_loss=m['log_band_mae']
    # Per-note velocity, onset, gate duration and deletion pass with real renderer feedback.
    for i in range(len(best)-1,-1,-1):
        current=best[i]; winner=best; win=best_loss
        variants=[replace(current,velocity=int(np.clip(current.velocity+d,1,127))) for d in [-10,10]]
        variants += [replace(current,start=round(current.start+d,3)) for d in [-.01,.01]]
        variants += [replace(current,duration=round(current.duration+d,3)) for d in [-.02,.02]]
        variants += [None]
        for variant in variants:
            candidate=best.copy()
            if variant is None: candidate.pop(i)
            else:
                if variant.start<0 or variant.duration<.02 or variant.start+variant.duration>seconds: continue
                if any(j!=i and n.pitch==variant.pitch and
                       max(n.start,variant.start)<min(n.start+n.duration,variant.start+variant.duration)-1e-7
                       for j,n in enumerate(best)): continue
                candidate[i]=variant
            a=piano.render(candidate,seconds+.5); loss=metrics(y,a)['log_band_mae']
            if loss<win: winner=candidate; win=loss
        best=winner; best_loss=win
        if i%40==0: print('renderer refine remaining',i,'loss',best_loss,flush=True)
    audio=render_saved(piano,best,out/'refined',seconds+.5)
    results['refined']=dict(metrics(y,audio),notes=len(best))
    repeated=piano.render(read_midi(out/'refined.mid'),seconds+.5)
    repeatability=dict(spectral_convergence=metrics(audio,repeated)['spectral_convergence'],
                       log_band_loss_delta=abs(metrics(y,repeated)['log_band_mae']-results['refined']['log_band_mae']))
    report=dict(input=str(Path(args.input).resolve()),seconds=seconds,renderer=config,
                results=results,render_repeatability=repeatability,pursuit=trace,baseline_velocity_trials=baseline_trials,velocity_trials=trials,wall_seconds=time.time()-started,
                caveats=['No human listening evaluation; auxiliary ASR, if run, is recorded separately in outputs/asr.json.',
                         'Magnitude addition is approximate; phase interactions evaluated only on re-render.',
                         '73 pitches, 3 velocities, 3 gate durations; short dictionary tails are truncated.',
                         'Baseline is heuristic and has a different note count.'])
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for ax,name in zip(axs,['target','baseline','dictionary','refined']):
        x,_=sf.read(out/f'{name}.wav'); z=magnitude(x)
        ax.imshow(20*np.log10(np.maximum(z,1e-4)),origin='lower',aspect='auto',
                  extent=[0,len(x)/SR,0,SR/2],vmin=-45,vmax=20,cmap='magma')
        ax.set_ylim(0,6000); ax.set_ylabel(name+'\nHz')
    axs[-1].set_xlabel('seconds'); fig.tight_layout(); fig.savefig(out/'spectrogram.png'); plt.close(fig)
    items=''.join(f'<h2>{name}</h2><audio controls src="{name}.wav"></audio>' for name in ['target','baseline','dictionary','refined'])
    (out/'listen.html').write_text('<!doctype html><meta charset="utf-8"><title>Talking MIDI experiment</title><h1>Fixed-piano speech fitting</h1>'+items+'<p>Same gain, no per-player normalization. Listen before reading the transcript.</p><img width="100%" src="spectrogram.png"><pre>'+json.dumps(results,indent=2)+'</pre>')
    piano.close(); print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()
