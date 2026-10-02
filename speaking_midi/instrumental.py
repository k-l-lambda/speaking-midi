"""Sparse, variable-duration piano approximation for instrumental recordings.

Unlike the speech-oriented dense fitter, each atom includes a complete held
note and release. Nonnegative convolutional regression chooses duration as
well as pitch and onset. A coefficient penalty discourages redundant attacks.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import mido
import torch
import torch.nn.functional as F

from .core import Note, Piano, SR, SFONT, load_audio, write_midi
from .dense import DenseFeatures


def save_midi(notes, path, seconds):
    """Keep the source timeline, including trailing silence after the last note."""
    write_midi(notes, path)
    midi = mido.MidiFile(path)
    track = midi.tracks[0]
    total_ticks = round(mido.second2tick(seconds, midi.ticks_per_beat, 500000))
    track[-1].time += max(0, total_ticks-sum(event.time for event in track))
    midi.save(path)


def dictionary(piano, device, cache, durations, nfft=2048):
    config = dict(revision=1, durations=list(durations), nfft=nfft, sr=SR,
                  soundfont=hashlib.sha256(Path(piano.path).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache) / f'instrumental-{key}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    pitches = list(range(21, 109))
    velocities = [1, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 127]
    atoms = [(p, d) for p in pitches for d in durations]
    if path.exists():
        with np.load(path) as data:
            kernels, calibration = data['kernels'], data['calibration']
    else:
        feature = DenseFeatures(device, nfft=nfft)
        frames = round((max(durations) + .44) * 100)
        kernels, calibration = [], []
        for pitch, duration in atoms:
            waves = [piano.render([Note(pitch, v, .04, duration)], frames*.01)
                     for v in velocities]
            base = waves[velocities.index(48)]
            kernels.append(feature(base).cpu().numpy()[:, :frames])
            energy = np.array([np.sum(w*w) for w in waves])
            calibration.append(energy / max(float(np.sum(base*base)), 1e-12))
        kernels, calibration = np.array(kernels), np.array(calibration)
        np.savez_compressed(path, kernels=kernels, calibration=calibration)
    return torch.tensor(kernels, device=device), calibration, atoms, velocities, config


def decode(values, calibration, atoms, velocities, seconds, grid=.04, min_velocity=12):
    """Select strongest nonoverlapping events per key, retaining full note gates.

    A single MIDI channel cannot unambiguously release overlapping instances of
    the same pitch. Resolve competing duration/onset hypotheses before export.
    """
    candidates = []
    threshold_index = velocities.index(min_velocity)
    for i, (pitch, duration) in enumerate(atoms):
        curve = np.maximum.accumulate(calibration[i])
        for frame in np.flatnonzero(values[i] >= curve[threshold_index]):
            onset = float(frame * grid)
            if onset >= seconds:
                continue
            velocity = int(np.clip(np.rint(np.interp(values[i, frame], curve, velocities)), 1, 127))
            candidates.append((float(values[i, frame])*duration, Note(
                pitch, velocity, onset, min(duration, seconds-onset))))
    accepted = {p: [] for p, _ in atoms}
    for _, note in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(note.start >= old.start+old.duration-1e-8 or
               old.start >= note.start+note.duration-1e-8 for old in accepted[note.pitch]):
            accepted[note.pitch].append(note)
    return sorted([n for group in accepted.values() for n in group], key=lambda n: (n.start, n.pitch))


def fit_segment(y, bank, calibration, atoms, velocities, *, device, steps=600,
                sparsity=.03, min_velocity=12, refit_steps=200):
    feature = DenseFeatures(device, nfft=2048)
    target = feature(y)[None]
    frames = target.shape[-1]
    values = torch.full((1, len(atoms), (frames+3)//4), .001,
                        device=device, requires_grad=True)
    optimizer = torch.optim.Adam([values], lr=.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.001)
    history = []
    target_log = torch.log1p(target/.001)
    for iteration in range(steps):
        optimizer.zero_grad()
        estimate = F.conv_transpose1d(values, bank, stride=4)[:, :, 4:4+frames]
        reconstruction = (torch.log1p(estimate.clamp_min(0)/.001)-target_log).square().mean()
        penalty = sparsity * values.sum() / frames
        loss = reconstruction + penalty
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            values.clamp_(0, 500)
        if iteration % 100 == 0 or iteration == steps-1:
            history.append(dict(step=iteration, spectral=float(reconstruction.detach()),
                                penalty=float(penalty.detach())))
    notes = decode(values.detach().cpu().numpy()[0], calibration, atoms, velocities,
                   len(y)/SR, min_velocity=min_velocity)
    # Selection removes competing atoms. Refit the surviving events together so
    # their velocities account for energy formerly assigned to discarded atoms.
    # The support stays fixed: this pass cannot create extra attacks.
    mask = torch.zeros_like(values)
    atom_index = {atom: i for i, atom in enumerate(atoms)}
    raw = values.detach().cpu().numpy()[0]
    for note in notes:
        # At the right boundary, the exported gate can be shorter than its atom.
        compatible = [d for p, d in atoms if p == note.pitch and
                      abs(min(d, len(y)/SR-note.start)-note.duration) < 1e-7]
        duration = max(compatible, key=lambda d: raw[atom_index[note.pitch, d], round(note.start/.04)]*d)
        mask[0, atom_index[note.pitch, duration], round(note.start/.04)] = 1
    with torch.no_grad():
        values.mul_(mask)
    optimizer = torch.optim.Adam([values], lr=.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(refit_steps, 1), eta_min=.001)
    for iteration in range(refit_steps):
        optimizer.zero_grad()
        estimate = F.conv_transpose1d(values*mask, bank, stride=4)[:, :, 4:4+frames]
        reconstruction = (torch.log1p(estimate.clamp_min(0)/.001)-target_log).square().mean()
        loss = reconstruction + sparsity*values.sum()/frames
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            values.clamp_(0, 500).mul_(mask)
    if refit_steps:
        history.append(dict(step=steps+refit_steps-1, stage='fixed-support refit',
                            spectral=float(reconstruction.detach())))
        notes = decode(values.detach().cpu().numpy()[0], calibration, atoms, velocities,
                       len(y)/SR, min_velocity=min_velocity)
    return notes, history


def fit_audio(y, piano, *, device='cpu', cache='outputs/cache', steps=600,
              sparsity=.03, durations=(.08, .24, .64), chunk_seconds=12., context=1.2):
    if steps < 1 or sparsity < 0 or chunk_seconds <= 0 or context < max(durations)+.4:
        raise ValueError('Invalid steps, sparsity, chunk length, or context')
    started = time.monotonic()
    bank, calibration, atoms, velocities, config = dictionary(piano, device, cache, durations)
    notes, reports = [], []
    chunk_samples = round(chunk_seconds*SR)
    context_samples = round(context*SR)
    for start in range(0, len(y), chunk_samples):
        end = min(len(y), start+chunk_samples)
        left, right = max(0, start-context_samples), min(len(y), end+context_samples)
        local, history = fit_segment(y[left:right], bank, calibration, atoms, velocities,
                                    device=device, steps=steps, sparsity=sparsity)
        for n in local:
            absolute = n.start + left/SR
            if start/SR-1e-7 <= absolute < end/SR-1e-7:
                notes.append(Note(n.pitch, n.velocity, absolute, min(n.duration, len(y)/SR-absolute)))
        reports.append(dict(start=start/SR, end=end/SR, history=history))
        print(f'Fitted {end/SR:.2f}/{len(y)/SR:.2f} s; {len(notes)} notes', flush=True)
    # Resolve same-key overlap across independently fitted chunk boundaries.
    notes.sort(key=lambda n: (n.start, n.pitch))
    previous = {}
    for note in notes:
        old = previous.get(note.pitch)
        if old is not None and old.start+old.duration > note.start:
            old.duration = note.start-old.start
        previous[note.pitch] = note
    return notes, dict(config=config, device=device, steps=steps, sparsity=sparsity,
                       chunk_seconds=chunk_seconds, context=context, refit_steps=200,
                       seconds=len(y)/SR, notes=len(notes), chunks=reports,
                       fitting_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--out', type=Path, required=True, help='Output .mid path; no WAV is rendered')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--sparsity', type=float, default=.03)
    parser.add_argument('--cache', default='outputs/cache')
    parser.add_argument('--soundfont', default=SFONT)
    parser.add_argument('--seconds', type=float, help='Fit only an initial excerpt')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists; choose a new path')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    y = load_audio(args.input)
    if args.seconds is not None:
        if args.seconds <= 0:
            parser.error('--seconds must be positive')
        y = y[:round(args.seconds*SR)]
    piano = Piano(args.soundfont)
    try:
        notes, report = fit_audio(y, piano, device=device, cache=args.cache,
                                  steps=args.steps, sparsity=args.sparsity)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_midi(notes, args.out, len(y)/SR)
        args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
        print(f'Saved {args.out}: {len(notes)} notes', flush=True)
    finally:
        piano.close()


if __name__ == '__main__':
    main()
