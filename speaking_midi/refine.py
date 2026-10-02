"""Renderer-in-the-loop experiments, preserving the original refinement as v1."""
from dataclasses import replace
import time
import numpy as np
from .core import SR, make_log_band_loss


def valid(notes, index, note, seconds):
    return (0 <= note.start and note.duration >= .02 and note.start+note.duration <= seconds
            and 36 <= note.pitch <= 108 and not any(j != index and n.pitch == note.pitch and
            max(n.start,note.start) < min(n.start+n.duration,note.start+note.duration)-1e-7
            for j,n in enumerate(notes)))


def refine(piano, y, notes, *, passes=1, pitch=False, min_improvement=0.0):
    """v1-compatible global velocity calibration followed by discrete local sweeps."""
    start=time.monotonic();seconds=len(y)/SR;loss=make_log_band_loss(y);evaluations=0
    def evaluate(candidate):
        nonlocal evaluations
        evaluations+=1
        return loss(piano.render(candidate,seconds+.5))
    best=list(notes);best_loss=evaluate(best);history=[best_loss];accepted=[]
    for delta in [-12,-6,6,12]:
        candidate=[replace(n,velocity=int(np.clip(n.velocity+delta,1,127))) for n in notes]
        score=evaluate(candidate)
        if score<best_loss-min_improvement:best=candidate;best_loss=score
    for sweep in range(passes):
        changes=0
        for i in range(len(best)-1,-1,-1):
            n=best[i];winner=best;score=best_loss;kind=None
            variants=[('velocity',replace(n,velocity=int(np.clip(n.velocity+d,1,127)))) for d in [-10,10]]
            variants += [('start',replace(n,start=round(n.start+d,3))) for d in [-.01,.01]]
            variants += [('duration',replace(n,duration=round(n.duration+d,3))) for d in [-.02,.02]]
            if pitch: variants += [('pitch',replace(n,pitch=n.pitch+d)) for d in [-2,-1,1,2]]
            variants += [('delete',None)]
            for label,v in variants:
                candidate=best.copy()
                if v is None:candidate.pop(i)
                else:
                    if not valid(best,i,v,seconds):continue
                    candidate[i]=v
                current=evaluate(candidate)
                if current<score-min_improvement:winner=candidate;score=current;kind=label
            best=winner;best_loss=score
            if kind:changes+=1;accepted.append(dict(pass_index=sweep,kind=kind,loss=score))
            if i%40==0:print('refine remaining',i,'loss',best_loss,flush=True)
        history.append(best_loss)
        print('refinement',sweep,'notes',len(best),'loss',best_loss,'changes',changes,flush=True)
        if not changes:break
    return best,dict(evaluations=evaluations,loss_history=history,accepted=accepted,
                     wall_seconds=time.monotonic()-start)
