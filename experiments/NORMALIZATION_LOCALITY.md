# Recording-wide normalization couples unrelated passages

Date: 2026-10-09.

The local instrumental fitter and the deployed Space at `04f9110` use identical
instrumental and dense source files. The actual SoundFont SHA-256 is identical
(`5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3`).
The Space fits a cropped prefix on CPU; the October 9 local batch fitted complete
recordings on CUDA. Both normalize the supplied recording to RMS 0.035 in the
previous implementation. This makes cropping a material algorithmic input change.

For the local `半岛铁盒.mp3`, the full decoded recording lasts 61.7796 seconds.
Before normalization, the first 30 seconds have RMS 0.10913, versus 0.23633 over
the complete decoded recording. Cropping first therefore raises the target
amplitude by **2.1656 times / 6.7117 dB** relative to the same passage in a full
fit. Fixed log-power floor, activation penalty, nonlinear velocity calibration
and export threshold make the optimization non-invariant to this scaling.
Normalizing each time window instead would change dynamics and introduce
window-dependent gains; it is not the correction used here.

## Controlled comparison

The user supplied `/home/camus/work/diary-job/temp/instrumental.mid` as the earlier
HF result. It lasts 30 seconds and contains 3,122 notes. Its exact job report is
not available, so not every setting or renderer-version difference can be proven.
The 600-step local full-recording result contains 6,044 notes in total.

Reproduced the Space's preprocessing locally: decode and average channels,
read only the first 30 seconds, resample, then normalize that prefix. Fit with
600 initial steps plus the fixed 200-step support refit and the same dictionary.

| MIDI | Notes before 30 s | Log-band MAE against the 30 s normalized target |
|---|---:|---:|
| Supplied HF result | 3,122 | 0.39459 |
| Local whole-recording fit, cropped for comparison | 2,179 | 0.52575 |
| Prefix-first reproduction, CUDA | 3,062 | 0.39913 |
| Prefix-first reproduction, CPU | 3,063 | 0.39911 |

The cropped MIDI measurement excludes note-on events at the exact cutoff with a
1e-6-second tolerance; a preliminary float-based count included four boundary
notes and reported 2,183. After gain matching, the FFmpeg-prepared and Space-style
input waveforms differ by about 0.396% relative L2. Locally cached dictionaries
from the two checkouts differ by about 0.0053% in kernel relative L2 and 0.0084%
in calibration relative L2. Neither CPU/CUDA nor these small local cache
differences reproduce the much larger global-level effect.

All MIDI comparisons were rendered with the same local Grand Piano, in memory.
High-key (pitch >=84) versus lower-key isolated RMS ratios were about 0.309 for
the supplied HF MIDI and 0.258 for the local whole-recording MIDI. This crude
pitch split does **not** establish worse noise SNR in the local output. High
notes are not all noise, and perceptual artifacts depend on timing and masking.
The experiment identifies an input-conditioning defect and substantially
reproduces the reference MIDI statistics; it does not fully explain subjective
noise quality. Raw local evidence is in `outputs/ban-dao-hf-diagnostic/`.

## Local research correction

`load_audio` now accepts `target_rms=None` and an explicit positive fixed `gain`.
The instrumental CLI defaults to `--normalization fixed --gain 1`, preserving
input amplitudes. `--normalization legacy-rms` remains available to reproduce
previous measurements. Input-level settings are recorded in the report. The
legacy speech loader default is retained, and the Space has not been changed by
this local research edit.

A regression test changes a distant tail's amplitude by 30 times and separately
truncates the file: the common prefix is bit-identical with fixed gain, while the
legacy RMS path changes it. The default fixed-gain path no longer has a
recording-wide level statistic. Local resampling support and the optimizer's
12-second intervals with 1.2-second context remain; arbitrary crop boundaries
can still affect the nearby fit. The test establishes preprocessing locality,
not complete invariance of all independently optimized windows.

Existing delivered MIDI files are unchanged. The earlier seven-file batch remains
paused, with three delivered files; the explicit 600-step replacement of
`半岛铁盒.mid` remains the prior legacy-normalized file until a new generation is
requested. A paused worker already loaded old code and must be restarted rather
than blindly resumed to use the corrected normalization.
