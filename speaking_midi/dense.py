"""Dense convolutional piano fitting without text or an ASR model in the optimizer."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from .core import Piano, Note, SR, SFONT, load_audio, render_saved, metrics


class DenseFeatures:
    def __init__(self, device, nfft=1024, power=2):
        self.device, self.nfft, self.power = device, nfft, power
        self.bank = torch.tensor(librosa.filters.mel(sr=SR, n_fft=nfft, n_mels=128,
                                                    fmin=40, fmax=11000), device=device)
        self.window = torch.hann_window(nfft, device=device)

    def __call__(self, y):
        x = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        spectrum = torch.stft(x, self.nfft, 240, window=self.window,
                              pad_mode='constant', return_complex=True).abs()
        return self.bank @ spectrum.pow(self.power)


def decode_activations(activation, calibration, pitches, velocities, gate, grid_frames, seconds):
    """Map per-pitch nonnegative energy coefficients to calibrated MIDI velocity."""
    notes = []
    for pi, pitch in enumerate(pitches):
        curve = np.maximum.accumulate(calibration[pi])
        for frame, amount in enumerate(activation[pi]):
            onset = frame * grid_frames * .01
            if onset + gate > seconds or amount < curve[2]:
                continue
            velocity = int(np.clip(np.rint(np.interp(amount, curve, velocities)), 1, 127))
            notes.append(Note(pitch, velocity, onset, gate))
    return notes


def fit_dense(y, piano, *, cache='outputs/cache', device=None, nfft=1024,
              power=2, gate=.02, grid_frames=2, velocity=48, floor=.001, steps=600):
    """Fit dense note strengths and quantize them; callers must render saved MIDI.

    No reference text, recognizer, pitch bend, source audio layer, or output EQ is
    used. Templates approximate additive mel energy, not the exact synthesizer.
    """
    if (steps < 1 or grid_frames < 1 or gate <= 0 or gate > grid_frames*.01+1e-9
            or not np.isfinite(floor) or floor <= 0 or power not in [1,2]
            or not 1 <= velocity <= 127 or nfft < 2 or len(y) == 0
            or not np.isfinite(y).all()):
        raise ValueError('Invalid dense fitting configuration or audio')
    started = time.monotonic()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    feature = DenseFeatures(device, nfft, power)
    pre, length = 4, 40
    pitches = list(range(21,109))
    velocities = [1,4,8,12,16,24,32,48,64,80,96,112,127]
    config = dict(power=power, gate=gate, velocity=velocity, sr=SR, nfft=nfft, hop=240,
                  pitches=pitches, soundfont_sha256=hashlib.sha256(Path(piano.path).read_bytes()).hexdigest(), revision=1)
    key = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache)/f'dense-{key}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    cache_hit = path.exists()
    if cache_hit:
        with np.load(path) as data:
            kernels, calibration = data['kernels'], data['calibration']
    else:
        kernels, calibration = [], []
        for pitch in pitches:
            base = piano.render([Note(pitch,velocity,pre*.01,gate)],length*.01)
            kernels.append(feature(base).cpu().numpy()[:,:length])
            levels = []
            for value in velocities:
                wave = piano.render([Note(pitch,value,pre*.01,gate)],length*.01)
                levels.append(float(np.sum(wave*wave)))
            calibration.append((np.array(levels)/max(float(np.sum(base*base)),1e-12))**(power/2))
        kernels, calibration = np.array(kernels), np.array(calibration)
        np.savez_compressed(path,kernels=kernels,calibration=calibration)
    dictionary_seconds = time.monotonic()-started
    kernel = torch.tensor(kernels,device=device)
    target = feature(y)[None]
    frames = target.shape[-1]
    count = (frames+grid_frames-1)//grid_frames
    activation = torch.full((1,len(pitches),count),.01,device=device,requires_grad=True)
    optimizer = torch.optim.Adam([activation],lr=.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.001)
    target_log = torch.log1p(target/floor)
    history = []
    for iteration in range(steps):
        optimizer.zero_grad()
        estimate = F.conv_transpose1d(activation,kernel,stride=grid_frames)[:,:,pre:pre+frames]
        loss = (torch.log1p(estimate.clamp_min(0)/floor)-target_log).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            activation.clamp_(min=0,max=500)
        if iteration%100 == 0 or iteration == steps-1:
            history.append([iteration,float(loss.detach())])
            print('dense',iteration,float(loss.detach()),flush=True)
    values = activation.detach().cpu().numpy()[0]
    notes = decode_activations(values,calibration,pitches,velocities,gate,grid_frames,len(y)/SR)
    report = dict(config=config, optimization=dict(steps=steps,grid_frames=grid_frames,floor=floor,
                  device=device,lr_schedule='cosine .04 to .001'), loss_history=history,
                  dictionary=dict(cache_hit=cache_hit,seconds=dictionary_seconds,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                  fitting_seconds=time.monotonic()-started)
    return notes, report, values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input')
    ap.add_argument('--out',type=Path,default=Path('outputs/dense-demo'))
    ap.add_argument('--soundfont',default=SFONT)
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--cache',type=Path,default=Path('outputs/cache'))
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = ap.parse_args()
    if args.steps < 1:ap.error('steps must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    y = load_audio(args.input)
    sf.write(args.out/'target.wav',y,SR,subtype='FLOAT')
    piano = Piano(args.soundfont)
    try:
        notes, report, activation = fit_dense(y,piano,steps=args.steps,cache=args.cache,
                                             device=None if args.device=='auto' else args.device)
        audio = render_saved(piano,notes,args.out/'dense',len(y)/SR+.5)
        np.savez_compressed(args.out/'activation.npz',activation=activation)
        report.update(metrics=metrics(y,audio),notes=len(notes),wall_seconds=time.monotonic()-started)
        (args.out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report['metrics'],indent=2))
    finally:piano.close()


if __name__ == '__main__':main()
