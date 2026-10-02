"""Write the ASR improvement report from audited dense MIDI artifacts."""
import json
from pathlib import Path
r=json.loads(Path('experiments/dense-results.json').read_text());rows=r['samples'];agg=r['aggregate']
def rate(method,language):return agg[method][language]['error_rate']
lines=['# Dense piano fitting: a material ASR improvement', '', 'Date: 2026-10-02.', '',
'This experiment changes the note representation and fitting procedure after the [balanced-objective experiment](BALANCED_FITTING.md). All five existing poetry recordings are evaluated at their original lengths. The output remains ordinary MIDI for bank 0 / program 0 of the same `gm.sf3` Grand Piano: no source-speech layer, pitch bend, effects, post-EQ, or waveform editing.', '',
'## Result', '',
f'Pooled Chinese CER falls from **{rate("original","Chinese"):.1%} to {rate("dense","Chinese"):.1%}**. Pooled English WER falls from **{rate("original","English"):.1%} to {rate("dense","English"):.1%}**. Both reductions exceed the exploratory material-improvement criterion of 20 percentage points, fixed before evaluating the three non-pilot recordings. Corpus totals pool character edits and reference characters within Chinese, and word edits and reference words within English; the two languages are not pooled into a single accuracy.', '',
'| Recording | Source CER/WER | Original piano | Balanced piano | Dense piano | Fresh-render range | Notes, original → dense |', '|---|---:|---:|---:|---:|---:|---:|']
for x in rows:
    a=x['asr'];lo=min(a[k]['error_rate'] for k in ['dense','repeat1','repeat2']);hi=max(a[k]['error_rate'] for k in ['dense','repeat1','repeat2'])
    lines.append(f'| {x["title"]} | {a["source"]["error_rate"]:.1%} | {a["original"]["error_rate"]:.1%} | {a["balanced"]["error_rate"]:.1%} | {a["dense"]["error_rate"]:.1%} | {lo:.1%}–{hi:.1%} | {x["original"]["notes"]} → {x["dense"]["notes"]} |')
lines += ['', '![ASR comparison](dense-asr.png)', '',
'The gain includes correctly recognized content, not only fewer hallucinated insertions. For example, the selected The Tyger MIDI is transcribed as:', '',
'> '+next(x['asr']['dense']['hypothesis'] for x in rows if x['name']=='the-tyger'), '',
'This remains imperfect: “Tyger” becomes “Tiger”, “What immortal” is lost, and “thy” becomes “by”. Chinese homophones and traditional/simplified character differences remain common. The corpus is small and uses one generated voice; this establishes an ASR improvement on these recordings, not general human intelligibility.', '',
'## Algorithm', '',
'The preceding optimizer searched a small collection of medium-duration notes and refined roughly 80–182 events. The new representation allows an onset for every one of the 88 piano keys on a 20 ms grid, with a 20 ms gate. Final clips contain 1,218–3,140 notes. This is a computer-played MIDI texture, not a score intended for a human pianist.', '',
'1. Render each pitch at reference velocity 48 using the actual piano. Use a 40 ms onset offset and a 400 ms waveform window, preserving attack, decay and release in each template.',
'2. Compute 128 Slaney-normalized mel bands from squared STFT magnitude, FFT 1024 and hop 240 at 24 kHz, covering 40–11,000 Hz. Retain 40 temporal frames per template. The target is normalized to the same RMS 0.035 as earlier experiments.',
'3. Jointly optimize nonnegative coefficients for every pitch and grid position through a transposed convolution. Use projected Adam for 600 iterations, learning rate 0.04 decaying to 0.001 with a cosine schedule, and clamp coefficients to [0, 500].',
'4. Map each coefficient to MIDI velocity using measured single-note waveform-energy ratios at velocities 1, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112 and 127. Interpolate the monotone calibration curve, omit coefficients below the velocity-8 threshold, and drop events extending beyond the input interval.',
'5. Export MIDI, reload it, and render with FluidSynth. Only that actual rendered audio is passed to ASR. The output includes the standard 0.5-second release tail.', '',
'With target mel power $M$, template $D$, pitch $p$, grid index $n$, and frame $t$, the selected surrogate is:', '',
'$$', r'\widehat M_{b,t}=\sum_{p,n}a_{p,n}D_{p,b,t+4-2n},\qquad a_{p,n}\geq0.', '$$', '',
'$$', r'L=\operatorname{mean}_{b,t}\left[\log\left(1+\widehat M_{b,t}/10^{-3}\right)-\log\left(1+M_{b,t}/10^{-3}\right)\right]^2.', '$$', '',
'The four-frame offset accounts for the template’s delayed note onset. This model adds mel powers rather than waveforms. It ignores phase interference, repeated-note voice interactions and some velocity-dependent timbre changes; it is a search surrogate. Actual MIDI rendering and ASR are therefore necessary validation, not optional postprocessing.', '',
'No reference text, transcript loss, ASR embedding or ASR gradient is used during fitting. ASR was used to compare pilot configurations. This differs from the earlier rejected quadrature prototype: all grid coefficients are jointly optimized instead of adding a short list of greedily ranked events. The experiments change density, templates and the objective together; they do not isolate density as the sole cause of the improvement.', '',
'## Pilot selection and validation', '',
'Seven configurations were explored on Quiet Night Thoughts and Sonnet 18. The 20 ms, power-domain, floor-0.001 configuration was chosen because it improved both languages and had the lowest mean of their two raw error rates. That mean was a pilot-selection heuristic, not a cross-language population accuracy. Parameters were then frozen for Spring Dawn, Stork Tower and The Tyger.', '',
'| Pilot configuration | Quiet Night CER | Sonnet 18 WER |', '|---|---:|---:|']
for name,p in r['pilots'].items():
    results={x['sample']:x['error_rate'] for x in p['asr']['results']}
    lines.append(f'| {name} | {results["quiet-night"]:.1%} | {results["sonnet-18"]:.1%} |')
lines += ['', 'The first power and magnitude pilots used constant learning rate 0.04; subsequent pilots used cosine decay. The full configuration and optimization histories are preserved in [dense-results.json](dense-results.json). The three validation recordings are from the same previously studied TTS corpus, not an independent test of new voices.', '',
'## Repeatability, decoding controls and CPU', '',
'Two fresh FluidSynth instances independently re-rendered the exported MIDI files. ASR was repeated on those WAV files. A second control disabled temperature fallback (`temperature=0`) for both source/original and dense audio. No text prompt, VAD, or previous-text conditioning was supplied; language was fixed, beam size was 5, and the same cached faster-whisper-base checkpoint ran on CPU int8.', '',
'| Evaluation | Pooled Chinese CER | Pooled English WER |', '|---|---:|---:|']
for key,label in [('original','Original'),('balanced','Balanced'),('dense','Dense selected'),('repeat1','Fresh render 1'),('repeat2','Fresh render 2'),('fixed_original','Original, temperature 0'),('fixed_dense','Dense, temperature 0'),('cpu','CPU fit, including CPU-built dictionary')]:
    lines.append(f'| {label} | {rate(key,"Chinese"):.1%} | {rate(key,"English"):.1%} |')
cpu=r['cpu']['cold']
lines += ['', f'The cold CPU Sonnet 18 run, including building the template/calibration cache on CPU, took **{cpu["wall_seconds"]:.2f} s**; dictionary construction accounted for {cpu["dictionary"]["seconds"]:.2f} s. All five recordings were subsequently fitted and recognized using the CPU path. The CUDA public CLI produced a byte-identical Sonnet 18 MIDI to the selected experiment. Timings are local observations, exclude ASR, and are not isolated throughput benchmarks.', '',
'With temperature fallback disabled, the original Spring Dawn output enters a repetitive hallucination (1110% CER), inflating the control’s pooled Chinese error. The headline improvement uses the standard-fallback baseline of 100%, not this inflated number. Dense results are unchanged in the temperature-zero check.', '',
'Raw character scoring retains traditional/simplified differences. English scoring lowercases and splits alphanumeric words; archaic “Tyger” spelling is scored literally. Error rates may exceed 100% because of insertions. The default ASR temperature fallback and tiny renderer/backend changes can alter wording, so the report retains all repetitions rather than selecting the best transcript.', '',
'## Acoustic tradeoffs', '',
'The dense model is selected for ASR evidence rather than the earlier log-band metric. It does not improve every acoustic metric:', '',
'| Recording | Original → dense log-band MAE | Original → dense linear SC | Dense per-clip time after dictionary setup |', '|---|---:|---:|---:|']
for x in rows:
    a=x['original'];b=x['dense']['metrics']
    lines.append(f'| {x["name"]} | {a["log_band_mae"]:.5f} → {b["log_band_mae"]:.5f} | {a["spectral_convergence"]:.4f} → {b["spectral_convergence"]:.4f} | {x["dense"]["wall_seconds"]:.2f} s |')
lines += ['', 'These results show why optimizing one spectral distance alone was insufficient. Better ASR is still a proxy: no blinded human study or independent recognizer was used, and familiar poems may benefit from language-model priors. The optimizer receives only audio, and parameters were frozen before the three validation clips, but broader claims require unfamiliar speech and additional voices.', '',
'## Reproduction and artifacts', '',
'Portable audio, MIDI and hashes are committed under [examples/poetry/dense](../examples/poetry/dense/README.md). Open its `index.html` locally for source/dense playback with hidden transcripts. The source clips are unchanged. For a new input:', '',
'```bash', 'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m speaking_midi.dense \\',
'  examples/poetry/quiet-night.wav --out outputs/dense-demo --device auto',
'# All steps, including dictionary creation, support --device cpu.', '```', '',
'To reproduce the selected corpus run and deterministic ASR check:', '',
'```bash', 'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.dense_fit \\',
'  --only quiet-night spring-dawn stork-tower sonnet-18 the-tyger',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.poetry_asr \\',
'  --root outputs/dense-floor-high --variants dense --temperature 0 \\',
'  --out outputs/dense-floor-high/asr-temperature0.json',
'OPENBLAS_NUM_THREADS=1 scripts/env.sh -m experiments.repeat_dense',
'```', '',
'`experiments/poetry_asr.py` requires the existing faster-whisper-base cache and the optional [ASR dependency](requirements.txt). It never downloads a model automatically. `experiments/package_dense.py` is an audit/packaging script for the complete experiment directory: it additionally requires archived pilot runs, both repeated ASR runs, the previous reports and CPU runs. It checks source/audio hashes, MIDI event types, note counts, ASR edit distances and material-improvement thresholds. A fresh rerun can differ slightly in notes/transcription because of floating-point and renderer state; the committed artifacts pin the reported measurements.', '',
'## Related primary sources', '',
'[Semi-Supervised Convolutive NMF for Automatic Piano Transcription](https://arxiv.org/abs/2202.04989) describes temporal template factorization for piano transcription. This experiment instead fixes templates from the renderer and fits speech; the paper does not establish this result. [Peter Ablinger’s Deus Cantando](https://ablinger.mur.at/txt_qu3god.html) is an artistic precedent for computer-controlled piano speech. The ASR model is [Whisper base](https://huggingface.co/openai/whisper-base), used without text prompts or fine-tuning.']
Path('experiments/DENSE_ASR.md').write_text('\n'.join(lines)+'\n')
asset=['# Dense piano speech examples','', 'These are the selected fixed-Grand-Piano MIDI reconstructions from the [dense ASR experiment](../../../experiments/DENSE_ASR.md). Source speech is in the parent directory. No speech audio is mixed into the reconstructions.','', 'Open [index.html](index.html) locally for offline playback; reveal the transcript only after listening. Each output has a 0.5-second release tail.','', '| Recording | Piano WAV | MIDI |','|---|---|---|']
for x in rows:
    name=x['name'];asset.append(f'| {x["title"]} | [Audio]({name}.wav) | [MIDI]({name}.mid) |')
asset += ['', 'Exact source, MIDI and WAV hashes and the frozen fitting settings are recorded in [manifest.json](manifest.json). ASR remains imperfect; these examples demonstrate an improvement on this corpus, not general intelligibility.']
Path('examples/poetry/dense/README.md').write_text('\n'.join(asset)+'\n')
print('Wrote dense ASR report and portable asset documentation.')
