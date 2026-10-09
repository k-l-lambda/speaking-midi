# Pitch-dependent timing experiment

The goal is to recover speech detail lost by a soft treble penalty while giving
high-register notes shorter key-down durations. This is an experimental local
fitter, not a change to the instrumental CLI or the deployed Space.

## Parameterization and objective

At 24 kHz, features use a common 240-sample (10 ms) hop and 128 mel bands. The
pitch partitions and MIDI gate candidates are:

| Register | MIDI pitches | Activation stride | Gates |
| --- | --- | --- | --- |
| Low | 21–59 | 4 frames | 80 / 240 / 640 ms |
| Middle | 60–83 | 2 frames | 40 / 80 / 240 ms |
| High | 84–108 | 1 frame | 10 / 20 / 40 ms |

For register $r$, $K_r$ is a dictionary rendered with the supplied grand-piano
SoundFont and $A_r$ contains nonnegative note strengths. Strided transposed
convolutions place each dictionary on the shared feature timeline:

$$
\widehat{Y}=\sum_r K_r *_{s_r} A_r,
\qquad
L=\operatorname{mean}_{f,t}
\left[\log(1+\widehat{Y}_{f,t}/10^{-3})-
\log(1+Y_{f,t}/10^{-3})\right]^2
+\frac{0.03}{T}\sum_r\lVert A_r\rVert_1.
$$

There is no extra cost for high pitch. Templates include the attack and release,
with 400 ms of post-gate context. An energy calibration across 13 MIDI velocities
maps fitted strengths to integer velocities. Initial projected Adam uses 600
steps; same-pitch overlap suppression and a minimum velocity of 12 select the
support, followed by 200 strength-refinement steps. Fit windows have 12-second
centers and 1.2-second context. MIDI uses one grand-piano channel; boundaries are
trimmed to prevent overlapping same-pitch notes.

Fixed input gain is 1. No mean, maximum, or RMS over the recording is used to
scale its samples or spectral bands. The log floor is constant. This removes the
previous dependence of local amplitude on distant loud passages. It does not
guarantee identical fitted notes after cropping: analysis context, finite
optimization, and chunk boundaries still matter.

## Separate analysis-window ablation

`--analysis single` uses FFT 2048 for every mel band and is the default.
`--analysis multiwindow` uses FFT 2048 below 1 kHz, 1024 from 1 to 3 kHz, and 512
above 3 kHz, selected by mel-band center. Each uses the same hop, and powers are
multiplied by $(2048/N)^2$ to align the nominal sinusoidal scale. Matching target
and dictionary transforms are essential. Frequency bands and MIDI pitch groups
are different concepts: harmonics of a low note also occupy high-frequency bands.

## Controlled pilot

`experiments.multigrid` compares four configurations on identical fixed-gain
inputs: the first 12 seconds of *Half Island Iron Box*, seconds 20–28 of the
decoded *Painted Heart* vocal reference, and 8.64 seconds of the generated
*Quiet Night Thought* poem. All use 600 initial + 200 refinement steps on CUDA.
Every score uses the actual exported-and-reloaded MIDI rendered by FluidSynth,
not the additive dictionary estimate. Rendered audio stays in memory.

| Excerpt | Configuration | Log-band MAE ↓ | 10 ms RMS-envelope correlation ↑ | Notes |
| --- | --- | ---: | ---: | ---: |
| Half Island Iron Box | Uniform instrumental | 0.5427 | 0.6439 | 1,327 |
| | Conservative + treble cost 0.4 | 0.6390 | 0.6412 | 1,059 |
| | Pitch-dependent timing | **0.4927** | **0.6699** | 5,023 |
| | Timing + shorter analysis windows | 0.4986 | 0.6477 | 5,174 |
| Painted Heart vocal | Uniform instrumental | 0.7285 | 0.7480 | 722 |
| | Conservative + treble cost 0.4 | 0.8361 | 0.7700 | 516 |
| | Pitch-dependent timing | **0.6797** | 0.8096 | 4,473 |
| | Timing + shorter analysis windows | 0.6846 | **0.8307** | 4,514 |
| Quiet Night Thought | Uniform instrumental | 0.6703 | 0.7890 | 805 |
| | Conservative + treble cost 0.4 | 0.7366 | **0.8123** | 692 |
| | Pitch-dependent timing | **0.6588** | 0.8079 | 3,183 |
| | Timing + shorter analysis windows | 0.6688 | 0.8071 | 3,462 |

Relative to uniform instrumental timing, the timing-only variant reduces
log-band MAE by 9.2%, 6.7%, and 1.7%. Mean high-register gates fall from
222/218/189 ms to 21/18/15 ms. This supports finer temporal representation; it
does not prove a perceptual improvement or accurate transcription.

The above-3-kHz excess-energy proxy decreases on the instrumental and poem
excerpts but slightly increases on the vocal excerpt (1.23e-5 to 1.50e-5).
It is not a direct measure of chime perception: notes beginning at MIDI 84 have
fundamentals around 1 kHz, below that evaluation band. The recovered spectrum
still has substantial underfit, and pitch-dependent timing does not dominate
every metric. Shorter analysis windows also show mixed results.

Raw metrics: [pilot results](multigrid-results.json).

## Speech-recognition cross-check

On the poem, cached `Systran/faster-whisper-base` runs on CPU/int8 with Chinese
language fixed, beam 5, temperature 0, no VAD, no prompt, and no conditioning on
previous text. Saved MIDI is rendered, then resampled to 16 kHz. Raw character
error counts use the existing project scorer, without converting traditional
Chinese characters to simplified Chinese. Thus these are reproducible diagnostic
scores, not a reliable estimate of human intelligibility.

| Input | Errors / 20 reference characters | Raw CER |
| --- | ---: | ---: |
| Original voice | 3 | 15% |
| Uniform instrumental | 20 | 100% |
| Conservative + treble cost 0.4 | 150 | 750% |
| Pitch-dependent timing | 17 | 85% |
| Timing + shorter analysis windows | 12 | 60% |
| Existing dense speech fitter, fixed gain | 9 | 45% |

The suppressed variant causes repetitive hallucinations; insertions allow CER
above 100%. The new grids improve this small ASR check relative to the
instrumental fits, but **do not outperform the existing dense speech fitter**.
That baseline uses 20 ms gates/onsets across every pitch, FFT 1024, no sparsity
penalty, and 600 steps without support refit. The experiment is therefore a
timing compromise for mixed material, not an established replacement for speech
mode. The baseline comparison is not an isolated timing ablation.

Exact hypotheses and model revision: [ASR results](multigrid-asr.json).

## Reproduction and limits

```bash
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.multigrid
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.multigrid_asr
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.multigrid_full
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m unittest discover -s tests -v
```

The full-candidate runner writes `outputs/multigrid/半岛铁盒-分音区时间粒度.mid`
and `outputs/multigrid/画心-前38秒-分音区时间粒度.mid`, and refuses to overwrite
existing files. The pilot runner overwrites its own named experiment outputs.
The full runner checks positive gates, same-pitch note closure, full MIDI duration,
finite actual renders, and absence of output clipping. CPU tests recover a
20 ms high note starting at 130 ms, between the old 40 ms grid positions. Feature
tests verify that changing distant samples does not change earlier bands.

Both full candidates passed validation: 61.779625 seconds / 29,057 notes for
*Half Island Iron Box*, and 38 seconds / 17,459 notes for *Painted Heart*. Peak
render amplitudes were 0.401 and 0.228, respectively. See the
[full-render results](multigrid-full-results.json). All 22 tests passed. No WAV
files were exported, and the previously paused batch remains paused.

This remains a dense sound approximation rather than a playable score. More
notes increase event-processing cost. MIDI note-off does not silence a piano
instantaneously, and repeated notes can accumulate release tails. Very short
gates may be interpreted differently by another synthesizer; evaluation here is
specific to the existing FluidSynth renderer and `gm.sf3`. Additive feature
templates cannot model all phase interference or velocity-dependent timbre.
Listening comparison is still required before choosing a production default.
