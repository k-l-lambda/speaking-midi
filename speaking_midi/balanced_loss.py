"""A fixed-scale spectral and envelope objective for cached note search."""
import numpy as np
import torch
import torch.nn.functional as F
from .core import SR, HOP, NFFT, filterbank


class BalancedLoss:
    """L_log + spectral_weight * SC + envelope_weight * (NRMSE + 1-corr).

    RMS envelopes use nonoverlapping 10 ms frames. No candidate normalization or
    alignment is applied. The same objective validates complete renderer passes.
    """
    def __init__(self, y, device=None, spectral_weight=.2, envelope_weight=.2):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.samples = len(y)
        self.spectral_weight = spectral_weight
        self.envelope_weight = envelope_weight
        self.window = torch.hann_window(NFFT, device=self.device)
        self.bank = torch.as_tensor(filterbank(), device=self.device)
        target = torch.as_tensor(y, dtype=torch.float32, device=self.device)[None]
        self.spectrum = self.stft(target)
        self.log_bands = torch.log1p(self.bank @ self.spectrum)
        self.spec_norm = self.spectrum.norm().clamp_min(1e-12)
        self.envelope = self.rms(target)
        self.env_norm = self.envelope.norm().clamp_min(1e-12)
        self.centered = self.envelope-self.envelope.mean(-1, keepdim=True)
        self.centered_norm = self.centered.norm().clamp_min(1e-12)

    def stft(self, x):
        return torch.stft(x, NFFT, HOP, window=self.window,
                          pad_mode='constant', return_complex=True).abs()

    def rms(self, x):
        padded = F.pad(x, (0, (-self.samples) % HOP))
        return padded.reshape(len(x), -1, HOP).square().mean(-1).sqrt()

    def components(self, audios):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(audios)[:, :self.samples],
                                dtype=torch.float32, device=self.device)
            spectrum = self.stft(x)
            log = (torch.log1p(self.bank @ spectrum)-self.log_bands).abs().mean((1, 2))
            sc = (spectrum-self.spectrum).flatten(1).norm(dim=1)/self.spec_norm
            envelope = self.rms(x)
            rms_error = (envelope-self.envelope).norm(dim=1)/self.env_norm
            centered = envelope-envelope.mean(-1, keepdim=True)
            corr = (centered*self.centered).sum(-1)/(centered.norm(dim=1)*self.centered_norm).clamp_min(1e-12)
            corr = corr.clamp(-1, 1)
            return {key: value.cpu().numpy() for key, value in
                    dict(log_band_mae=log, spectral_convergence=sc,
                         envelope_nrmse=rms_error, envelope_correlation=corr).items()}

    def __call__(self, audios):
        c = self.components(audios)
        return (c['log_band_mae'] + self.spectral_weight*c['spectral_convergence']
                + self.envelope_weight*(c['envelope_nrmse']+1-c['envelope_correlation']))
