"""Compare register-dependent grids on voice and instrumental controls."""
import json
from pathlib import Path
import numpy as np
from speaking_midi.core import load_audio,SR,Piano,read_midi
from speaking_midi.instrumental import fit_audio,save_midi
from speaking_midi.multigrid import fit_multigrid
from experiments.chime_objective import measure,SOURCE

OUT=Path('outputs/multigrid');OUT.mkdir(parents=True,exist_ok=True)

def main():
    results=[]
    cases=[('bandao',SOURCE/'半岛铁盒.mp3',0,12),
           ('huaxin-voice',Path('footages/画心.wma'),20,8),
           ('quiet-night',Path('examples/poetry/quiet-night.wav'),0,8.64)]
    for name,source,start,seconds in cases:
        # WMA is unavailable to libsndfile; use the previously decoded input.
        if source.suffix=='.wma':source=Path('outputs/footages/画心/input.wav')
        y=load_audio(source,target_rms=None)[round(start*SR):round((start+seconds)*SR)]
        for mode in ['legacy','suppressed','multigrid','multiwindow']:
            p=Piano()
            try:
                if mode in ['legacy','suppressed']:
                    ns,report=fit_audio(y,p,device='cuda',steps=600,objective='legacy' if mode=='legacy' else 'conservative',treble_cost=0 if mode=='legacy' else .4)
                else:ns,report=fit_multigrid(y,p,device='cuda',steps=600,analysis='single' if mode=='multigrid' else 'multiwindow')
                path=OUT/f'{name}-{mode}.mid';save_midi(ns,path,len(y)/SR)
                ns=read_midi(path);actual=p.render(ns,len(y)/SR+.5)
                row=dict(case=name,variant=mode,**measure(y,actual,ns))
                row['registers']={}
                for register,lo,hi in [('low',21,60),('mid',60,84),('high',84,109)]:
                    group=[n for n in ns if lo<=n.pitch<hi]
                    row['registers'][register]=dict(notes=len(group),mean_gate=float(np.mean([n.duration for n in group])) if group else 0)
                results.append(row)
                (OUT/'results.json').write_text(json.dumps(results,indent=2));(OUT/f'{name}-{mode}.report.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(row),flush=True)
            finally:p.close()
if __name__=='__main__':main()
