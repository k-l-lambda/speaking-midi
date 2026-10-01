# First experiment results — September 30, 2026

With the fixed `gm.sf3` Grand Piano preset, temporal piano templates followed by actual-renderer refinement reduced spectral error compared with the project's spectral-peak baseline. **Reliable speech intelligibility has not been demonstrated.** Whisper transcribed the original TTS recordings correctly, but largely failed to recover the piano versions. No human blind-listening evaluation was performed.

## Samples and configuration

- Hardware: NVIDIA RTX 3090. Speech model: Qwen3-TTS-12Hz-0.6B-CustomVoice, revision `85e237c12c027371202489a0ec509ded67b5e4b5`. Both weight files were checked against the SHA256 values published in Hugging Face metadata.
- Chinese: Vivian, “你好，今天的天气真好。” (“Hello, the weather is really nice today.”), 2.96 seconds. English: Ryan, “Hello, can you hear the piano speaking?”, 3.20 seconds. Generation used seed 42, SDPA, and bfloat16.
- Renderer: FluidSynth 2.2.5 at 24 kHz, bank 0/program 0 (**Grand Piano**), gain 0.8, reverb and chorus disabled. SoundFont SHA256: `5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3`.
- Search: 657 rendered temporal templates, a 10 ms onset grid, and a maximum of 240 pursuit iterations. Both samples stopped before that limit. One actual-renderer refinement pass adjusted velocity, onset, and key-hold duration, and considered note deletion.
- The baseline also received a global velocity search. All methods used the same target RMS without separate output loudness normalization. Reported reconstruction metrics came from rendering MIDI files read back after export.

See [README.md](README.md#algorithm-details) for the objective functions, dictionary construction, pursuit derivation, and refinement procedure.

## Spectral measurements

Both distances are lower-is-better. The baseline and template search have different note budgets, so this is not an equal-budget comparison.

| Sample | Method | Notes | Log-band MAE | Linear STFT spectral convergence |
|---|---|---:|---:|---:|
| Chinese | Spectral-peak baseline | 170 | 0.439872 | 0.809863 |
| Chinese | Temporal piano dictionary | 52 | 0.376894 | 0.774738 |
| Chinese | Actual-renderer refinement | 50 | 0.358844 | 0.753684 |
| English | Spectral-peak baseline | 215 | 0.384901 | 0.811547 |
| English | Temporal piano dictionary | 72 | 0.347858 | 0.728330 |
| English | Actual-renderer refinement | 66 | 0.319011 | 0.722481 |

Final log-band error fell by **18.4% for Chinese and 17.1% for English** relative to the calibrated baseline. Refinement reduced this error by another **4.8% and 8.3%**, respectively, relative to the initial dictionary solution. The spectrograms still show substantial mismatch associated with piano harmonics and release tails, including incomplete reconstruction of high-frequency speech structure.

Repeating the final MIDI rendering changed the target log-band loss by `7.42e-6` for Chinese and `3.25e-6` for English, much less than the differences between methods. FluidSynth's first-voice and block-boundary behavior, together with int16 dithering, prevents a claim of sample-for-sample determinism. Metrics cover only the input duration; the extra 0.5-second tail retained in each output WAV is excluded from the loss.

## Auxiliary speech recognition

Evaluation used the locally cached faster-whisper-base model with CPU int8 inference, no text prompt, beam size 5, VAD disabled, and previous-text conditioning disabled. Chinese scoring retains Unicode alphanumeric characters, removes punctuation, and does not convert between traditional and simplified characters. English scoring uses lowercase words. One small ASR model cannot replace blind listening, and its transcripts on piano audio may be hallucinations.

| Sample and metric | Original TTS | Baseline | Dictionary | Refined |
|---|---:|---:|---:|---:|
| Chinese CER | 0% | 100% | 100% | 88.9% |
| English WER | 0% | 100% | 100% | 85.7% |

The refined Chinese audio was incorrectly transcribed as “請按旁邊的按鈕” (“Please press the button beside it”); the English output yielded only “Hello?”. The numerical improvement in Chinese comes from incidental character overlap and should not be interpreted as meaningful intelligibility improvement.

The supported conclusion is limited: accounting for the fixed instrument reduced the selected spectral distances on these two samples. The experiment does not establish reliable recovery of the spoken sentences.

## Artifacts and validation

The artifact links below refer to locally generated files in `outputs/`, which is excluded from Git. Run the commands in the README to generate them in a fresh checkout.

- Chinese: [listening page](outputs/zh/listen.html), [MIDI](outputs/zh/refined.mid), [piano WAV](outputs/zh/refined.wav).
- English: [listening page](outputs/en/listen.html), [MIDI](outputs/en/refined.mid), [piano WAV](outputs/en/refined.wav).
- Records: [TTS generation manifest](outputs/tts/manifest.json), [Chinese metrics](outputs/zh/metrics.json), [English metrics](outputs/en/metrics.json), [complete ASR results](outputs/asr.json).
- [Setup and algorithm explanation](README.md).

Two automated tests passed: MIDI timing/velocity round-trip with rendering repeatability, and recovery of a known single note's pitch, velocity, and duration. Additional checks on all three MIDI/WAV variants for both speech samples verified note counts, valid pitch and velocity ranges, absence of overlapping same-key holds, finite audio values, and peaks below 1. A separate three-note piano control experiment also completed; piano self-reconstruction does not establish speech intelligibility.

The next evaluation should include listening without transcripts and held-out samples. Algorithm experiments should examine lower velocities, denser duration grids, multiple spectral resolutions, repeated coordinate refinement, and explicit treatment of tail spillover and unvoiced consonants. ASR should remain an auxiliary measurement rather than a fixed recognizer to overfit.
