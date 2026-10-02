"""Block-aligned note caching and batched spectral coordinate descent.

FluidSynth is almost additive with effects disabled and unclipped output. Cached
single notes propose local edits cheaply; a full MIDI render validates each pass.
"""
from collections import OrderedDict
from dataclasses import replace
import time
import numpy as np
import torch
from .core import SR,HOP,NFFT,Note,features,filterbank,make_log_band_loss,metrics
from .refine import valid


class NoteCache:
    def __init__(self,piano,limit=4096):
        self.piano=piano;self.cache=OrderedDict();self.limit=limit;self.misses=0

    def waveform(self,note):
        sample=round(note.start*SR);phase=sample%64;duration=round(note.duration*SR)
        key=(note.pitch,note.velocity,phase,duration)
        if key not in self.cache:
            self.misses+=1
            self.cache[key]=self.piano.render([Note(note.pitch,note.velocity,phase/SR,duration/SR)],
                                             (phase+duration)/SR+.8)
            if len(self.cache)>self.limit:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return sample-phase,self.cache[key]

    def add(self,audio,note,gain=1.):
        start,wave=self.waveform(note);end=min(len(audio),start+len(wave))
        if end>start:audio[start:end]+=gain*wave[:end-start]

    def render(self,notes,seconds):
        result=np.zeros(round(seconds*SR),np.float32)
        for n in notes:self.add(result,n)
        return result


class BatchLogLoss:
    def __init__(self,y,device=None):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.samples=len(y)
        self.target=torch.as_tensor(np.log1p(features(y)),device=self.device)
        self.bank=torch.as_tensor(filterbank(),device=self.device)
        self.window=torch.hann_window(NFFT,device=self.device)

    def __call__(self,audios):
        x=torch.as_tensor(np.asarray(audios)[:,:self.samples],dtype=torch.float32,device=self.device)
        with torch.no_grad():
            z=torch.stft(x,NFFT,HOP,window=self.window,pad_mode='constant',return_complex=True).abs()
            return (torch.log1p(self.bank@z)-self.target).abs().mean((1,2)).cpu().numpy()


def refine_cached(piano,y,notes,*,passes=2,pitch=True,device=None,objective=None):
    """Refine notes; a custom batched objective also validates real-render passes.

    ``verified_loss_history`` records that selected objective, not necessarily
    log-band MAE. The objective must expose a ``device`` attribute.
    """
    started=time.monotonic();seconds=len(y)/SR;best=list(notes);cache=NoteCache(piano)
    custom_objective=objective is not None
    objective=objective or BatchLogLoss(y,device)
    true_loss=(lambda audio: float(objective([audio])[0])) if custom_objective else make_log_band_loss(y)
    audio=piano.render(best,seconds+.5);loss=true_loss(audio)
    approximation=cache.render(best,seconds+.5)
    fidelity=metrics(audio,approximation)
    history=[loss];pass_reports=[];candidates_evaluated=0
    for sweep in range(passes):
        before=list(best);before_audio=audio.copy();before_loss=loss;edits={}
        for i in range(len(best)-1,-1,-1):
            n=best[i];vstep=10 if sweep==0 else 5;tstep=.02 if sweep==0 else .01;dstep=.04 if sweep==0 else .02
            variants=[('velocity',replace(n,velocity=int(np.clip(n.velocity+d,1,127)))) for d in [-vstep,vstep]]
            variants += [('start',replace(n,start=round(n.start+d,4))) for d in [-tstep,tstep]]
            variants += [('duration',replace(n,duration=round(n.duration+d,4))) for d in [-dstep,dstep]]
            if pitch:variants += [('pitch',replace(n,pitch=n.pitch+d)) for d in [-2,-1,1,2]]
            variants += [('delete',None)]
            labels=[('keep',n)];waves=[audio]
            removed=audio.copy();cache.add(removed,n,-1)
            for label,v in variants:
                if v is not None and not valid(best,i,v,seconds):continue
                a=removed.copy()
                if v is not None:cache.add(a,v)
                waves.append(a);labels.append((label,v))
            scores=objective(waves);k=int(np.argmin(scores));candidates_evaluated+=len(waves)
            if k and scores[k]<scores[0]-1e-6:
                label,v=labels[k];audio=waves[k]
                if v is None:best.pop(i)
                else:best[i]=v
                edits[label]=edits.get(label,0)+1
        verified=piano.render(best,seconds+.5);verified_loss=true_loss(verified)
        proposal_loss=float(objective([audio])[0]);accepted=verified_loss<before_loss-1e-6
        if accepted:audio=verified;loss=verified_loss
        else:best=before;audio=before_audio;loss=before_loss
        history.append(loss)
        pass_reports.append(dict(pass_index=sweep,accepted=accepted,edits=edits,
                                 proposed_loss=proposal_loss,verified_loss=verified_loss))
        print('cached refine',pitch,sweep,'loss',loss,'accepted',accepted,'edits',edits,flush=True)
    return best,dict(objective=type(objective).__name__,device=objective.device,wall_seconds=time.monotonic()-started,
        cache_misses=cache.misses,candidates_evaluated=candidates_evaluated,initial_additivity=fidelity,
        verified_loss_history=history,passes=pass_reports)
