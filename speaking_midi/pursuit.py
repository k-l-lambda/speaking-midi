"""Experimental log-band pursuit with a quadrature-magnitude mixing surrogate."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from .core import Note, SR, HOP, features


def fit_log_power(y, atoms, params, max_notes=240, shortlist=64, device=None):
    """Rank additions by a smoothed gradient, accept by exact surrogate log MAE.

    sqrt(sum(atom_magnitude**2)) models incoherent mixing, not phase interaction.
    The actual FluidSynth re-render is always needed to evaluate exported MIDI.
    """
    started=time.monotonic();device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    target=torch.as_tensor(features(y),device=device)
    kernels=torch.as_tensor(atoms,device=device).square()
    length=kernels.shape[-1];frames=target.shape[-1]
    target_log=F.pad(torch.log1p(target),(0,length))
    prediction=torch.zeros_like(target_log)
    mask=F.pad(torch.ones_like(target),(0,length))
    allowed=torch.as_tensor(np.arange(frames)[None]*HOP/SR+params[:,2,None]<=len(y)/SR+1e-8,device=device)
    scale=kernels.sum((1,2)).sqrt().clamp_min(1e-6)
    notes=[];history=[]
    with torch.no_grad():
        for iteration in range(max_notes):
            current=torch.log1p(prediction.sqrt())
            # Stabilized derivative of squared log magnitude, only for shortlisting.
            root=prediction.clamp_min(.01).sqrt()
            gradient=(target_log-current)/(root*(1+root))*mask
            scores=F.conv1d(gradient[None],kernels)[0,:,:frames]/scale[:,None]
            scores.masked_fill_(~allowed,-float('inf'))
            _,indices=scores.flatten().topk(min(shortlist,scores.numel()))
            k=indices//frames;t=indices%frames
            old=prediction.unfold(1,length,1)[:,t,:].permute(1,0,2)
            desired=target_log.unfold(1,length,1)[:,t,:].permute(1,0,2)
            valid=mask.unfold(1,length,1)[:,t,:].permute(1,0,2)
            old_error=(desired-torch.log1p(old.sqrt())).abs()
            new_error=(desired-torch.log1p((old+kernels[k]).sqrt())).abs()
            gain=((old_error-new_error)*valid).sum((1,2))
            gain.masked_fill_(~allowed[k,t],-float('inf'))
            improvement,selected=gain.max(0)
            if improvement.item()<=1e-5:break
            ki=int(k[selected]);ti=int(t[selected]);pitch,velocity,duration=params[ki]
            notes.append(Note(int(pitch),int(velocity),ti*HOP/SR,float(duration)))
            prediction[:,ti:ti+length]+=kernels[ki]
            for j in np.flatnonzero(params[:,0]==pitch):
                left=max(0,ti-int(np.ceil(params[j,2]*SR/HOP))+1)
                right=min(frames,ti+int(np.ceil(duration*SR/HOP)))
                allowed[j,left:right]=False
            history.append(float(improvement)/(target.numel()))
    return notes,dict(device=device,accepted=len(notes),surrogate_improvements=history,
                     shortlist=shortlist,wall_seconds=time.monotonic()-started)
