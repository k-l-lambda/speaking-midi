"""Experimental pitch-dependent note grids and frequency-dependent STFT windows."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from .core import Note,Piano,SR,SFONT,load_audio
from .dense import DenseFeatures
from .instrumental import decode,save_midi
from .instrumental_loss import InstrumentalLoss

REGISTERS = [dict(name='low', pitches=list(range(21,60)), grid=4, gates=(.08,.24,.64)),
             dict(name='mid', pitches=list(range(60,84)), grid=2, gates=(.04,.08,.24)),
             dict(name='high',pitches=list(range(84,109)),grid=1,gates=(.01,.02,.04))]
VELOCITIES = [1,4,8,12,16,24,32,48,64,80,96,112,127]


class MultiWindowFeatures:
    def __init__(self, device):
        self.device=device
        centers=librosa.mel_frequencies(n_mels=130,fmin=40,fmax=11000)[1:-1]
        self.parts=[]
        for nfft,mask in [(2048,centers<1000),(1024,(centers>=1000)&(centers<3000)),(512,centers>=3000)]:
            bank=librosa.filters.mel(sr=SR,n_fft=nfft,n_mels=128,fmin=40,fmax=11000)[mask]
            self.parts.append((nfft,torch.tensor(bank,device=device),torch.hann_window(nfft,device=device)))

    def __call__(self,y):
        x=torch.as_tensor(y,device=self.device,dtype=torch.float32)
        bands=[]
        for nfft,bank,window in self.parts:
            power=torch.stft(x,nfft,240,window=window,pad_mode='constant',return_complex=True).abs().square()
            # Align sinusoidal power scale to the original 2048-point analysis.
            bands.append((bank@power)*(2048/nfft)**2)
        return torch.cat(bands,dim=0)


def make_banks(piano,device,cache,analysis):
    feature=MultiWindowFeatures(device) if analysis=='multiwindow' else DenseFeatures(device,nfft=2048)
    config=dict(revision=1,registers=REGISTERS,analysis=analysis,sr=SR,hop=240,
                soundfont=hashlib.sha256(Path(piano.path).read_bytes()).hexdigest())
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    path=Path(cache)/f'multigrid-{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    cached=np.load(path) if path.exists() else None
    result=[];stored={}
    try:
        for i,group in enumerate(REGISTERS):
            atoms=[(p,d) for p in group['pitches'] for d in group['gates']]
            if cached is not None:kernels,calibration=cached[f'k{i}'],cached[f'c{i}']
            else:
                frames=round((max(group['gates'])+.44)*100)
                kernels=[];calibration=[]
                for pitch,gate in atoms:
                    waves=[piano.render([Note(pitch,v,.04,gate)],frames*.01) for v in VELOCITIES]
                    base=waves[VELOCITIES.index(48)]
                    kernels.append(feature(base).cpu().numpy()[:,:frames])
                    calibration.append(np.array([np.sum(w*w) for w in waves])/max(float(np.sum(base*base)),1e-12))
                kernels,calibration=np.array(kernels),np.array(calibration)
                stored[f'k{i}'],stored[f'c{i}']=kernels,calibration
            result.append((group,atoms,torch.tensor(kernels,device=device),calibration))
    finally:
        if cached is not None:cached.close()
    if stored:np.savez_compressed(path,**stored)
    return feature,result,config


def select_support(values,calibration,atoms,grid,seconds):
    raw=values.detach().cpu().numpy()[0]
    notes=decode(raw,calibration,atoms,VELOCITIES,seconds,grid=grid*.01,min_velocity=12)
    support=torch.zeros_like(values);indices={atom:i for i,atom in enumerate(atoms)}
    for n in notes:
        ds=[d for p,d in atoms if p==n.pitch and abs(min(d,seconds-n.start)-n.duration)<1e-7]
        frame=round(n.start/(grid*.01))
        d=max(ds,key=lambda d:raw[indices[n.pitch,d],frame]*d)
        support[0,indices[n.pitch,d],frame]=1
    return support


def segment(y,feature,banks,device,steps=600,refit_steps=200):
    target=feature(y)[None];frames=target.shape[-1]
    values=[torch.full((1,len(atoms),(frames+group['grid']-1)//group['grid']),.001,
                       device=device,requires_grad=True) for group,atoms,_,_ in banks]
    masks=[torch.ones_like(v) for v in values]
    criterion=InstrumentalLoss(target,[],objective='legacy')
    empty=torch.zeros(1,0,1,device=device)
    history=[]
    for phase,count in [('joint',steps),('refit',refit_steps)]:
        optimizer=torch.optim.Adam(values,lr=.04)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(count,1),eta_min=.001)
        for iteration in range(count):
            optimizer.zero_grad()
            estimate=sum(F.conv_transpose1d(v*mask,bank,stride=g['grid'])[:,:,4:4+frames]
                         for v,mask,(g,_,bank,_) in zip(values,masks,banks))
            spectral,_=criterion(estimate,empty,0.)
            loss=spectral+.03*sum(v.sum() for v in values)/frames
            loss.backward();optimizer.step();scheduler.step()
            with torch.no_grad():
                for v,mask in zip(values,masks):v.clamp_(0,500).mul_(mask)
        history.append(dict(phase=phase,spectral=float(spectral.detach())))
        if phase=='joint':
            masks=[select_support(v,c,atoms,g['grid'],len(y)/SR) for v,(g,atoms,_,c) in zip(values,banks)]
            with torch.no_grad():
                for v,mask in zip(values,masks):v.mul_(mask)
    notes=[]
    for v,(g,atoms,_,calibration) in zip(values,banks):
        notes+=decode(v.detach().cpu().numpy()[0],calibration,atoms,VELOCITIES,len(y)/SR,
                      grid=g['grid']*.01,min_velocity=12)
    return sorted(notes,key=lambda n:(n.start,n.pitch)),history


def fit_multigrid(y,piano,*,device='cpu',analysis='single',steps=600,cache='outputs/cache'):
    if (analysis not in {'single','multiwindow'} or steps<1 or len(y)==0
            or np.ndim(y)!=1 or not np.isfinite(y).all()):
        raise ValueError('Invalid analysis, steps, or audio')
    started=time.monotonic();feature,banks,config=make_banks(piano,device,cache,analysis)
    notes=[];histories=[];size=12*SR;context=round(1.2*SR)
    for start in range(0,len(y),size):
        end=min(start+size,len(y));left=max(0,start-context);right=min(len(y),end+context)
        local,history=segment(y[left:right],feature,banks,device,steps)
        for n in local:
            absolute=n.start+left/SR
            if start/SR-1e-7<=absolute<end/SR-1e-7:
                notes.append(Note(n.pitch,n.velocity,absolute,min(n.duration,len(y)/SR-absolute)))
        histories.append(dict(start=start/SR,end=end/SR,history=history))
        print(f'Multigrid {end/SR:.2f}/{len(y)/SR:.2f} s; {len(notes)} notes',flush=True)
    notes.sort(key=lambda n:(n.start,n.pitch));previous={}
    for n in notes:
        old=previous.get(n.pitch)
        if old is not None and old.start+old.duration>n.start:old.duration=n.start-old.start
        previous[n.pitch]=n
    return notes,dict(config=config,steps=steps,refit_steps=200,device=device,notes=len(notes),
                       seconds=len(y)/SR,history=histories,fitting_seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--steps',type=int,default=600)
    p.add_argument('--analysis',choices=['single','multiwindow'],default='single')
    p.add_argument('--gain',type=float,default=1.);p.add_argument('--soundfont',default=SFONT)
    args=p.parse_args()
    if args.out.exists():p.error('Output exists; choose a new path')
    y=load_audio(args.input,target_rms=None,gain=args.gain);piano=Piano(args.soundfont)
    try:
        notes,report=fit_multigrid(y,piano,device=args.device,steps=args.steps,analysis=args.analysis)
        args.out.parent.mkdir(parents=True,exist_ok=True);save_midi(notes,args.out,len(y)/SR)
        report['input_level']=dict(normalization='fixed',gain=args.gain)
        args.out.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    finally:piano.close()
if __name__=='__main__':main()
