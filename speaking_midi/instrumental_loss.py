"""Local spectral objectives for conservative instrumental piano fitting."""
import librosa
import math
import torch


class InstrumentalLoss:
    def __init__(self, target, atoms, *, objective='legacy', treble_cost=0.):
        if objective not in {'legacy', 'conservative'} or not math.isfinite(treble_cost) or treble_cost < 0:
            raise ValueError('Invalid instrumental objective or treble cost')
        self.target = target
        self.objective = objective
        self.treble_cost = treble_cost
        # Every scale is computed within one time frame. A distant loud passage
        # must not change this frame's target or weighting.
        self.floor = .001 if objective == 'legacy' else .001+.02*target.mean(dim=1, keepdim=True)
        self.target_log = torch.log1p(target/self.floor)
        self.scale = torch.sqrt(target.square().mean(dim=1, keepdim=True))+.001
        frequencies = librosa.mel_frequencies(n_mels=130, fmin=40, fmax=11000)[1:-1]
        self.high_weight = 1+2*torch.tensor((frequencies/3000).clip(0, 1),
                                          device=target.device, dtype=target.dtype)[None, :, None]
        # Soft cost rather than deleting a register. Real high notes can still
        # be selected when their contribution to reconstruction justifies it.
        self.pitch_cost = torch.tensor([max(0., (p-72)/12)**2 for p, _ in atoms],
                                       device=target.device, dtype=target.dtype)[None, :, None]

    def __call__(self, estimate, activation, sparsity):
        error = torch.log1p(estimate.clamp_min(0)/self.floor)-self.target_log
        spectral = error.square().mean()
        if self.objective == 'conservative':
            excess = (self.high_weight*error.clamp_min(0).square()).mean()
            linear = ((estimate-self.target)/self.scale).square().mean()
            spectral = spectral+.6*excess+.15*linear
        frames = self.target.shape[-1]
        penalty = sparsity*activation.sum()/frames
        penalty = penalty+self.treble_cost*(activation*self.pitch_cost).sum()/frames
        return spectral, penalty
