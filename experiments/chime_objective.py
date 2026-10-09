"""Ablate spectral overshoot and soft treble costs; save MIDI, not WAV output."""
import json
from pathlib import Path
import numpy as np
import librosa
from speaking_midi.core import load_audio,SR,Piano,read_midi,metrics
from speaking_midi.dense import DenseFeatures
from speaking_midi.instrumental import fit_audio,save_midi

OUT=Path('outputs/chime-objective');OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/home/camus/work/wechat-articles/general-piano/footages')

def measure(target, audio, notes):
    feature=DenseFeatures('cpu',nfft=2048)
    actual=audio[:len(target)]
    a,b=feature(target).numpy(),feature(actual).numpy()
    hz=librosa.mel_frequencies(n_mels=130,fmin=40,fmax=11000)[1:-1]
    high=hz>=3000;low=hz<3000
    denom=max(float(np.sum(a)),1e-12)
    frames=len(target)//240
    x=np.sqrt(np.mean(target[:frames*240].reshape(-1,240)**2,axis=1))
    z=np.sqrt(np.mean(actual[:frames*240].reshape(-1,240)**2,axis=1))
    return dict(metrics=metrics(target,audio),notes=len(notes),high_notes=sum(n.pitch>=84 for n in notes),
                high_excess=float(np.maximum(b[high]-a[high],0).sum()/denom),
                low_sc=float(np.linalg.norm(b[low]-a[low])/max(np.linalg.norm(a[low]),1e-12)),
                high_energy_ratio=float(b[high].sum()/max(a[high].sum(),1e-12)),
                envelope_correlation=float(np.corrcoef(x,z)[0,1]),
                rms_ratio=float(np.sqrt(np.mean(actual**2)/np.mean(target**2))))


def main():
    results=[]
    for song,start in [('半岛铁盒',0),('半岛铁盒',20),('枉凝眉',0)]:
        full=load_audio(SOURCE/(song+'.mp3'),target_rms=None)
        y=full[start*SR:(start+12)*SR]
        for label,objective,cost in [('baseline','legacy',0.),('conservative','conservative',0.),('treble010','conservative',.1),('treble040','conservative',.4)]:
            p=Piano()
            try:
                notes,report=fit_audio(y,p,device='cuda',steps=600,objective=objective,treble_cost=cost)
                path=OUT/f'{song}-{start}-{label}.mid';save_midi(notes,path,len(y)/SR)
                decoded=read_midi(path);audio=p.render(decoded,len(y)/SR+.5)
                row=dict(song=song,start=start,variant=label,objective=objective,treble_cost=cost,**measure(y,audio,decoded))
                results.append(row)
                (OUT/f'{song}-{start}-{label}.report.json').write_text(json.dumps(report,indent=2))
                (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
                print(json.dumps(row,ensure_ascii=False),flush=True)
            finally:p.close()
if __name__=='__main__':main()
