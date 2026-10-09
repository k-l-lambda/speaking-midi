# Reducing chime-like piano artifacts without recording-wide normalization

Date: 2026-10-09.

The user reported prominent wind-chime-like noise and requested an objective
change, also emphasizing that distant audio must not rescale local fitting.
This experiment retains the fixed Grand Piano, 80/240/640 ms gates, 40 ms grid,
600 initial iterations, and 200 support-refit iterations. It changes the spectral
objective and adds an optional soft high-pitch activation cost. It does not
filter the final waveform or delete a register after fitting.

## Input levels and time locality

Every candidate uses the same source amplitude: load the original MP3 with
channel averaging, resample to 24 kHz, and apply fixed gain 1. There is no
recording-wide or per-window waveform RMS normalization. Old generated MIDI used
legacy normalization, so a comparison directly against those files would confound
objective and level changes. The controls below re-fit the old objective on the
same fixed-gain input as every candidate.

Reading the original MP3 also avoids a preprocessing ambiguity: FFmpeg's default
stereo-to-mono matrix can scale amplitude differently from arithmetic channel
averaging. Legacy RMS normalization hid such gain differences. The new runs use
the core loader consistently for all variants.

Regression checks multiply a distant waveform tail by 30 and compare a cropped
prefix: fixed-gain samples remain identical. Another test multiplies distant
spectrogram frames by 100: earlier objective floors, scales, and gradients remain
identical. The existing 12-second optimization intervals and 1.2-second context
still permit nearby boundary effects; these tests do not claim arbitrary window
partition invariance.

## Objective

Let $M$ be target mel power, $P=D*A$ the piano-template reconstruction, and $B=128$.
For each time frame separately, define

$$
s_t=0.001+0.02\operatorname{mean}_b M_{b,t},\qquad
r_t=0.001+\sqrt{\operatorname{mean}_b M_{b,t}^2},
$$

$$
e_{b,t}=\log(1+P_{b,t}/s_t)-\log(1+M_{b,t}/s_t).
$$

The conservative loss is

$$
L=\operatorname{mean}e^2
+0.6\operatorname{mean}\left(w_b\max(e,0)^2\right)
+0.15\operatorname{mean}\left((P-M)/r_t\right)^2
+\frac{0.03}{T}\sum A
+\frac{\lambda_h}{T}\sum_{p,d,n}c(p)A_{p,d,n},
$$

where $w_b=1+2\min(f_b/3000,1)$ and
$c(p)=\max((p-72)/12,0)^2$. Frequencies are mel-filter centers. The second term
makes overprediction more expensive than equal log-domain underprediction;
the third protects stronger spectral components. The last term makes higher
piano pitches more expensive rather than forbidding them. It can also attenuate
valid high melody notes, so it is a deliberate preference, not source separation.

Neither $s_t$ nor $r_t$ uses other time frames, and neither rescales the input
waveform. Their role is local objective weighting. The coefficient penalty is
not an exact note-count penalty. Both optimization passes use the same objective.

## Comparisons

The first round compared the legacy objective, conservative spectral terms only,
and conservative terms plus pitch cost 0.1 on two twelve-second Hua Xin excerpts
(0–12 and 20–32 s) and Wang Ning Mei (0–12 s). A second round tested cost 0.4 on
the same excerpts after inspecting the first results. These are exploratory,
reused pilots, not an independent evaluation set.

All reported audio is produced by saving MIDI, reading it back, and rendering
through FluidSynth with the same SoundFont. Renders stay in memory; no WAV is
exported. To quantify register balance, render pitches below 84 and at/above 84
separately and measure their RMS. The upper group begins at C6 (about 1047 Hz).
It is not identical to a high-frequency audio band, and its energy is not all
noise. Reported dB changes are acoustic proxies for the user's complaint.

| Excerpt | Variant | High notes (pitch >=84) | High-register RMS change | Lower-register RMS change | Spectral convergence ↓ | Log-band MAE ↓ |
|---|---|---:|---:|---:|---:|---:|
| Hua Xin 0–12 s | Legacy | 776 | 0 dB | 0 dB | 0.7944 | 0.5427 |
| | Conservative only | 708 | -1.17 dB | -0.16 dB | 0.7878 | 0.5760 |
| | Pitch cost 0.1 | 538 | -2.54 dB | -0.11 dB | 0.7890 | 0.6088 |
| | Pitch cost 0.4 | 353 | -4.54 dB | -0.06 dB | 0.7918 | 0.6390 |
| Hua Xin 20–32 s | Legacy | 772 | 0 dB | 0 dB | 0.8120 | 0.6977 |
| | Pitch cost 0.1 | 394 | -4.39 dB | +0.17 dB | 0.7997 | 0.7911 |
| | Pitch cost 0.4 | 242 | -7.12 dB | +0.17 dB | 0.8002 | 0.8009 |
| Wang Ning Mei 0–12 s | Legacy | 525 | 0 dB | 0 dB | 0.8193 | 0.3595 |
| | Pitch cost 0.1 | 247 | -1.04 dB | +0.39 dB | 0.7982 | 0.3797 |
| | Pitch cost 0.4 | 140 | -2.89 dB | +0.40 dB | 0.8010 | 0.3910 |

The stronger setting reduces high-register energy while largely preserving
lower-register energy. Linear spectral convergence improves on these pilots,
but the older log-band measure worsens. This is a real tradeoff, not a uniform
reconstruction improvement. Energy above 3 kHz was already under-reconstructed
relative to the source, so further reducing that band alone cannot establish
better overall fidelity. No blind listening study or noise ground truth exists.

A known single piano note at pitch 96, velocity 80 remains detectable with cost
0.1, though the fit includes extra weak notes and does not recover the exact
original gate. The soft cost does not constitute a blanket pitch ban, but high
notes are not guaranteed to retain their original strength or duration.

## Implementation, evidence and reproduction

- `speaking_midi/instrumental_loss.py`: legacy/conservative losses and soft pitch cost.
- `speaking_midi/instrumental.py`: objective and pitch-cost options in both passes,
  reports, and CLI; fixed-gain normalization option from the preceding diagnosis.
- `tests/test_instrumental_loss.py`: distant-frame gradient independence,
  asymmetric overprediction cost, pitch cost, and CPU piano-note recovery.
- `experiments/chime_objective.py`: four-way pilots; `chime_register_audit.py`:
  isolated-register energy measurements. [Full numerical results](chime-objective-results.json)
  contain all twelve configurations and metrics.

The complete suite passed 20 tests. The existing multi-file batch remains paused.
The hosted Space has not been changed by this research experiment. The default
CLI objective remains `legacy`; the new objective is opt-in pending listening.

```bash
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.chime_objective
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.chime_register_audit
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m speaking_midi.instrumental \
  '/home/camus/work/wechat-articles/general-piano/footages/半岛铁盒.mp3' \
  --out outputs/chime-objective/new.mid --device cuda --steps 600 \
  --normalization fixed --gain 1 --objective conservative --treble-cost 0.4
```

The local runs and full-recording reports are under `outputs/chime-objective/`.
Both softer (0.1) and stronger (0.4) full-length MIDI variants are provided for
listening; existing delivered MIDI remains unchanged. These files preserve source
amplitude with a fixed gain, so their overall loudness need not match older
RMS-normalized outputs. Match listening loudness when comparing those older files.

## Full-length files delivered

Both MIDI files cover the complete 61.779625-second source and use fixed input
gain 1, 600 initial steps, and 200 refit steps:

| Local output in the article's footages directory | Pitch cost | Notes | Notes at/above C6 | Actual render peak |
|---|---:|---:|---:|---:|
| 半岛铁盒-抑制高音.mid | 0.1 | 7,109 | 2,054 | 0.3258 |
| 半岛铁盒-强抑制高音.mid | 0.4 | 6,241 | 1,241 | 0.3197 |

Both were reloaded and checked for complete note releases, positive gate lengths,
source duration, absence of same-key overlap, and finite real rendering without
clipping. Full render buffers were discarded rather than saved as WAV.
[Hashes and settings](chime-full-results.json) identify the delivered files.
The existing `半岛铁盒.mid` hash is unchanged, and the prior multi-file batch
processes were verified to remain stopped.
