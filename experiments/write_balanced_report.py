"""Generate the follow-up report from measured artifacts."""
import json
from pathlib import Path
import numpy as np
r = json.loads(Path('experiments/balanced-results.json').read_text())
rows = r['samples']
def mean(variant, key):
    return float(np.mean([x[variant][key] for x in rows]))
a, b = mean('old', 'log_band_mae'), mean('balanced', 'log_band_mae')
lines = ['# Balanced spectral and envelope fitting', '', 'Date: 2026-10-02.', '',
'This follows the [five-poem backtest](POETRY_BACKTEST.md). The previous cached optimizer reduced log-band error but worsened envelope correlation on every recording. This experiment changes the objective while retaining the same cached waveform search, four passes, candidate edits, fixed target RMS, SoundFont, and initial quick MIDI files.', '',
'## Outcome', '',
f'Across the same five full recordings, mean log-band MAE decreases from **{a:.5f} to {b:.5f} ({100*(1-b/a):.2f}%)** relative to the original full-render optimizer. Mean linear spectral convergence decreases from **{mean("old", "spectral_convergence"):.4f} to {mean("balanced", "spectral_convergence"):.4f}**; mean envelope correlation increases from **{mean("old", "envelope_correlation"):.4f} to {mean("balanced", "envelope_correlation"):.4f}**. Each recording improves on all three measures. These are corpus averages, not confidence intervals.', '',
'The balanced method gives back some log-band improvement compared with the previous log-only cached method. Multi-resolution log error increases slightly on three recordings. The result is a better measured compromise, not proof of better perceived quality or intelligible speech.', '',
'| Recording | Log MAE: original → balanced | Linear SC: original → balanced | Envelope correlation: original → balanced |',
'|---|---:|---:|---:|']
for x in rows:
    p, q = x['old'], x['balanced']
    lines.append(f'| {x["title"]} | {p["log_band_mae"]:.5f} → {q["log_band_mae"]:.5f} | {p["spectral_convergence"]:.4f} → {q["spectral_convergence"]:.4f} | {p["envelope_correlation"]:.4f} → {q["envelope_correlation"]:.4f} |')
lines += ['', '## Objective and implementation', '',
'Let $S$ be the linear STFT magnitude at FFT size 2048 / hop 240, and $E$ the RMS envelope in nonoverlapping 240-sample (10 ms) frames. The final envelope frame is zero-padded. All comparisons use the input interval; saved audio retains the original 0.5-second release tail. No candidate loudness normalization or time alignment is applied.', '',
'$$', r'L = L_{\mathrm{logband}} + \lambda_s\frac{\|S(\hat y)-S(y)\|_F}{\|S(y)\|_F} + \lambda_e\left(\frac{\|E(\hat y)-E(y)\|_2}{\|E(y)\|_2} + 1-\rho(E(\hat y),E(y))\right).', '$$', '',
'The selected weights are $\lambda_s=\lambda_e=0.05$. The envelope NRMSE term penalizes amplitude mismatch; correlation emphasizes timing and shape. Using correlation alone would allow incorrect overall amplitude. Linear spectral convergence counterbalances the log objective’s emphasis on quieter frequency bins. Weights are tied to the existing signal calibration and feature definition; they are not universal perceptual constants.', '',
'Candidate evaluation still subtracts/adds cached single-note waveforms, with block-phase alignment and batched GPU or CPU STFTs. Every complete pass is re-rendered through FluidSynth and accepted only if this **combined objective** improves. The original log-only pass guard is not imposed on balanced runs; individual metrics may trade off. MIDI is exported, reloaded and rendered before final scoring. Three further full renders per sample are recorded in the JSON to expose renderer variability.', '',
'## Parameter selection', '',
'The five weights 0.025, 0.05, 0.1, 0.2 and 0.4 were explored on Quiet Night Thoughts and Sonnet 18. Both spectral and envelope weights move together in this sweep. The selection is exploratory: after inspecting those pilots, 0.05 was chosen because 0.025 barely restored Quiet Night’s envelope correlation, while stronger weights sacrificed more log-band fidelity. The choice was recorded before evaluating Spring Dawn, Stork Tower and The Tyger. Those three are an internal validation set from the same previously evaluated TTS corpus, not independent held-out voices.', '',
'| Pilot | Weight | Log MAE | Linear SC | Envelope correlation |', '|---|---:|---:|---:|---:|']
for name, variants in r['pilots'].items():
    for key, v in variants.items():
        lines.append(f'| {name} | {v["weight"]:g} | {v["log_band_mae"]:.5f} | {v["spectral_convergence"]:.4f} | {v["envelope_correlation"]:.4f} |')
lines += ['', '## Ablation at weight 0.05', '',
'All variants use the same quick initialization and four cached passes. Log-only is the previous experiment; spectral-only and envelope-only retain log-band loss and add just the indicated term. These ablations were run on the two pilots after selecting the joint weight; they did not select another weight for the validation recordings.', '',
'| Pilot | Added terms | Log MAE | Linear SC | Envelope correlation |', '|---|---|---:|---:|---:|']
for name in r['selection']['pilot_samples']:
    row = next(x for x in rows if x['name'] == name)
    for label, v in [('None', row['log_only']), ('Linear spectrum', r['ablation']['spectral'][name]),
                     ('Envelope', r['ablation']['envelope'][name]), ('Both', row['balanced'])]:
        lines.append(f'| {name} | {label} | {v["log_band_mae"]:.5f} | {v["spectral_convergence"]:.4f} | {v["envelope_correlation"]:.4f} |')
lines += ['', 'Adding the envelope term is responsible for most of the temporal-envelope improvement. Adding linear spectral convergence improves that metric further in these pilots. A different objective changes the discrete search trajectory, so these gains are not additive.', '',
'## Tradeoffs and cost', '',
'| Recording | Log-only → balanced log MAE | Original → balanced multi-resolution log MAE | Balanced refinement time |', '|---|---:|---:|---:|']
for x in rows:
    lines.append(f'| {x["name"]} | {x["log_only"]["log_band_mae"]:.5f} → {x["balanced"]["log_band_mae"]:.5f} | {x["old"]["mr_log_mae"]:.5f} → {x["balanced"]["mr_log_mae"]:.5f} | {x["balanced"]["trace"]["wall_seconds"]:.1f} s |')
cpu = r['cpu']
lines += ['', f'The full Sonnet 18 CPU refinement took **{cpu["trace"]["wall_seconds"]:.1f} s**: log MAE {cpu["log_band_mae"]:.5f}, linear SC {cpu["spectral_convergence"]:.4f}, envelope correlation {cpu["envelope_correlation"]:.4f}. This excludes dictionary construction, initialization and TTS. GPU/CPU rounding can lead to different accepted notes. Timings were collected on the local RTX 3090 workstation with concurrent experiments and are observational, not isolated performance benchmarks.', '',
'![RMS envelope comparison](balanced-envelopes.png)', '',
'## ASR and remaining limitations', '',
'The selected five WAV files were transcribed with the same unprompted faster-whisper-base CPU int8 protocol used previously. Each transcription records the exact audio SHA-256. Chinese scores are CER and English scores are WER; insertions can make either exceed 100%.', '',
'| Recording | Balanced CER/WER |', '|---|---:|']
for x in rows:
    lines.append(f'| {x["name"]} | {x["asr"]["error_rate"]:.1%} |')
lines += ['', 'Transcriptions remain incorrect or empty. Incidental matching words and lower insertion counts do not demonstrate intelligibility. No blind human listening study has been performed. Five recordings from one synthetic voice cannot establish generalization. Envelope correlation is now an optimized quantity, not independent evidence of perceptual improvement. Multi-resolution error remains an auxiliary metric and exposes some of the tradeoff.', '',
'## Reproduction and usage', '',
'First reproduce the existing quick MIDI initializations using the preceding report. Then run:', '',
'```bash',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.balanced_search --weights .025 .05 .1 .2 .4',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.balanced_search --weights .05 --only spring-dawn stork-tower the-tyger',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.balanced_search --component spectral --weights .05',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.balanced_search --component envelope --weights .05',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.balanced_search --device cpu --weights .05 --only sonnet-18 --out outputs/balanced-cpu',
'scripts/env.sh -m experiments.select_balanced',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.poetry_asr --root outputs/balanced-selected --variants balanced',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.package_balanced',
'scripts/env.sh -m experiments.write_balanced_report',
'```', '',
'For another input, the experimental CLI exposes the combined objective explicitly:', '',
'```bash', 'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m speaking_midi.optimize \\',
'  examples/poetry/quiet-night.wav --out outputs/balanced-demo \\',
'  --passes 4 --objective balanced --balance-weight .05 --device auto',
'# Use --device cpu to force CPU pursuit and refinement.', '```', '',
'The CLI default remains the log-only experimental optimizer. [balanced-results.json](balanced-results.json) contains every pilot weight, the ablations, per-sample metrics, ASR transcripts and CPU results. Open `outputs/balanced-selected/index.html` for source / original / log-only / balanced playback and MIDI downloads. The HF demo is unchanged.', '',
'## Next research questions', '',
'Further tuning these five clips is unlikely to establish speech intelligibility. A useful next stage is phoneme-level or auditory-band envelope evaluation with a larger independent voice corpus and blinded listeners. Within this renderer, test finer low-velocity templates and residual-driven note insertion against the frozen balanced baseline, while reporting a complexity budget and all acoustic tradeoffs. A multi-resolution term could target the remaining regression, but adding more correlated losses should be evaluated on new data before choosing another default.']
Path('experiments/BALANCED_FITTING.md').write_text('\n'.join(lines)+'\n')
print(f'Mean log MAE {a:.5f} -> {b:.5f} ({100*(1-b/a):.2f}% reduction)')
