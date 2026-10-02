# Sparse variable-duration fitting for instrumental recordings

Date: 2026-10-02.

The speech-oriented dense fitter produced 98,705 notes for the 136.59-second
local recording `圣女之歌.mp3`, and 400,015 for the 432.08-second decoded
`画心.wma`. The user reported a noise-like texture. The new experiment reduces
repeated attacks while retaining a spectral approximation of each recording.
It continues to use the fixed `gm.sf3` Grand Piano; source audio is never mixed
into the MIDI reconstruction.

## Method

See the [README algorithm description](../README.md#instrumental-audio-sparse-variable-duration-fitting)
and `speaking_midi/instrumental.py`. The main differences from dense speech
fitting are three gate lengths (80/240/640 ms), a 40 ms onset grid, a 2048-point
FFT, and an L1 activation penalty. Single-note templates include their releases
and have independent pitch/duration velocity-energy calibration. Competing
same-key events are resolved before a 200-step fixed-support strength refit.
The second pass is important: simply discarding overlapping candidates loses
energy and can substantially increase reconstruction error.

The initial pass uses 600 projected-Adam steps, learning rate 0.04 decreasing to
0.001. The strength-refit pass uses the same endpoints over 200 steps. Both
minimize log-mel-power error plus activation sum divided by frame count and
weighted by `--sparsity`. Velocity 12 is the export threshold. This penalty is
not a strict density budget: note selection and quantization can make note count
nonmonotonic in its weight.

## Pilot evidence

Both pilots use the interval 20–28 seconds of the original recordings, normalized
with the full recording's RMS. Baseline events are cropped from the previously
exported full dense MIDI. Each candidate MIDI is written, read back, and rendered
in memory by FluidSynth; no pilot WAV is saved. Metrics exclude the release tail.
The pilot is small and selected from the same recordings used for final fitting.

| Recording / method | Notes in 8 s | Mean gate, ms | Spectral convergence ↓ | Log-band MAE ↓ |
|---|---:|---:|---:|---:|
| 画心 / dense | 2,706 | 20 | 0.81659 | 0.28445 |
| 画心 / variable duration, penalty 0.03 | 473 | 138 | 0.80073 | 0.27122 |
| 画心 / variable duration, penalty 0.10 | 468 | 138 | 0.80013 | 0.27154 |
| 圣女之歌 / dense | 4,547 | 20 | 0.79093 | 0.30526 |
| 圣女之歌 / variable duration, penalty 0.03 | 522 | 205 | 0.78828 | 0.31501 |
| 圣女之歌 / variable duration, penalty 0.10 | 520 | 210 | 0.78479 | 0.31578 |

Selected penalty **0.03**: it retains slightly more events and has the better
log-band error among the new variants on both excerpts. Attack counts fall by
82.5% and 88.5% versus the dense baseline. Spectral convergence improves slightly
on both excerpts; log-band error improves for 画心 but worsens about 3.2% for
圣女之歌. Before fixed-support refitting, the latter's log-band error was about
0.399, demonstrating the energy-loss problem. Minor baseline renderer variation
between passes is retained in the raw local evidence.

These measurements establish a much less attack-dense representation, not a
measured reduction in perceived noise. No listening study has been performed.
Longer gates do not guarantee faithful reproduction of sustained non-piano
instruments, and the model still approximates waveform interference and
velocity-dependent timbre through additive mel powers. It is not ordinary
polyphonic score transcription. Pitch or onset accuracy is not measured because
no reference score is available.

## Long-recording generation and checks

Full recordings are processed in 12-second central intervals with 1.2 seconds of
context on each side. Only central onsets are retained; same-key overlaps across
boundaries are shortened before MIDI export. Context limits local edge effects
but does not make independent windows globally consistent. The final right-edge
support mapping preserves the original duration atom even if its exported gate
is truncated to the input boundary.

```bash
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.instrumental_pilot
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m speaking_midi.instrumental \
  outputs/footages/圣女之歌/input.wav \
  --out footages/圣女之歌-器乐优化.mid --device cuda --sparsity 0.03
OPENBLAS_NUM_THREADS=1 scripts/env.sh -m speaking_midi.instrumental \
  outputs/footages/画心/input.wav \
  --out footages/画心-器乐优化.mid --device cuda --sparsity 0.03
```

The decoded inputs were reused from the earlier run; original MP3/WMA and original
MIDI files are preserved. The original MP3 decoder reported one invalid-data
warning but produced its full declared duration; this experiment uses that same
decoded input. MIDI and fit reports are generated without full-length WAV output.
Local input recordings and MIDI derivatives are not packaged as public examples.

`tests/test_instrumental.py` checks conflicting duration hypotheses, clipping and
MIDI release behavior, and recovery of a known sustained piano note on CPU. The
complete suite passed 16 tests before the final duration-support boundary fix;
the instrumental tests were rerun afterward. Pilot metrics are preserved in
[instrumental-pilot.json](instrumental-pilot.json); detailed histories and full-run
logs remain under `outputs/instrumental-pilot` and `outputs/instrumental-full`.

## Completed full-length MIDI outputs

| Recording | Duration including trailing silence | Dense notes | New notes | Reduction | Mean new gate |
|---|---:|---:|---:|---:|---:|
| 圣女之歌 | 136.5945 s | 98,705 | 12,446 | 87.39% | 205 ms |
| 画心 | 432.0770 s | 400,015 | 51,252 | 87.19% | 208 ms |

New files are `footages/圣女之歌-器乐优化.mid` and
`footages/画心-器乐优化.mid`. Original MIDI files are unchanged. Both new exports
were read back and checked for positive gates, input-duration bounds, matching
note counts, and absence of overlapping same-key events. End-of-track padding
preserves the source duration, including the quiet tail where no notes survive
selection. Full-run reports were moved from beside the MIDI files into
`outputs/instrumental-full/`; no full-length WAV was generated. Final checksums
and counts are in [instrumental-full.json](instrumental-full.json).

The two full fits took approximately 38 s and 117 s on the local RTX 3090,
excluding final export verification and using a cached template dictionary.
These are observational timings. The final instrumental tests passed after
adding source-duration preservation. The new mode and its controls are local
research changes; the Hugging Face demo was not changed by this experiment.
