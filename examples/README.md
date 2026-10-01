# Speaking MIDI demos

Two complete Qwen3-TTS 1.7B VoiceDesign recordings are fitted with the fixed `gm.sf3` Grand Piano preset. Open [index.html](index.html) from a local checkout for audio players and an interactive MIDI piano roll. Playback and visualization work offline. Playing one track pauses the others; clicking the piano roll seeks within the fitted piano audio.

| Language | Voice | Speech duration | Piano WAV duration |
|---|---|---:|---:|
| English | VoiceDesign | 7.84 s | 8.34 s |
| Chinese | VoiceDesign | 7.60 s | 8.10 s |

The piano files include 0.5 seconds of release tail. These are naturally generated utterances: the speech was not cropped, padded, or time-stretched to hit the target duration.

## Voice design

Both samples pass the following exact instruction to `generate_voice_design`:

> 一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。

English translation: A gentle, confident young woman with a clear voice, a warm tone, and a light, engaging delivery.

The model is [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign). It designs a voice from the description instead of selecting the Ryan or Vivian presets used in the earlier short experiment. The installed 0.6B CustomVoice interface ignores style instructions, so the generator rejects that combination rather than silently losing the requested description.

## Files

Each language directory contains:

- `source.wav`: original Qwen3-TTS output, 24 kHz mono, floating-point WAV.
- `target.wav`: the same recording at the fitter's common RMS level of 0.035.
- `baseline`, `dictionary`, and `refined`: `.wav`, `.mid`, and decoded note `.json` for each method.
- `spectrogram.png`: target and all three reconstructions, with common display limits.
- `piano-roll.png`: the final exported MIDI, with note velocity encoded by color.
- `metrics.json`: measured distances, renderer identity, configuration, and repeatability.

`prompts.json` contains the texts and exact voice instructions. `manifest.json` records model provenance, measured sample durations, generation settings, and SHA256 hashes for the per-language assets. The HTML page embeds the final MIDI note events so it can open locally without fetching JSON or loading any external scripts.

The static piano roll uses a viridis velocity color scale, while the browser roll uses blue-to-yellow with a labeled velocity legend. Both show MIDI key-hold intervals, not the full audible decay.

## Measured spectral fit

| Language | Method | Notes | Log-band MAE | Linear STFT convergence |
|---|---|---:|---:|---:|
| English | Baseline | 457 | 0.457734 | 0.831107 |
| English | Dictionary | 159 | 0.410655 | 0.759557 |
| English | Refined | 148 | 0.387474 | 0.760693 |
| Chinese | Baseline | 445 | 0.447968 | 0.831025 |
| Chinese | Dictionary | 123 | 0.411163 | 0.770461 |
| Chinese | Refined | 117 | 0.392300 | 0.798604 |

Final log-band MAE is 15.3% lower than the baseline for English and 12.4% lower for Chinese. Refinement optimizes log-band MAE; the linear STFT metric is reported separately and need not move in the same direction.

## Evaluation scope

These longer samples have not received human intelligibility or ASR evaluation. Lower spectral error does not establish that the piano speaks understandable words. Recognition scores in the repository's `RESULTS.md` concern the earlier, shorter recordings only. All spectral metrics use the same target calibration and exclude the extra release tail beyond the speech duration. The baseline and fitted versions have different note counts.

## Regenerate

To repeat fitting using the exact committed speech recordings, run from the repository root:

```bash
scripts/env.sh -m speaking_midi examples/en/source.wav --out outputs/demos/en
scripts/env.sh -m speaking_midi examples/zh/source.wav --out outputs/demos/zh
scripts/env.sh -m scripts.package_examples --tts examples --fits outputs/demos
```

The last command updates the committed demonstration assets and their hashes. It preserves the source speech and its generation provenance.

To synthesize new speech from the same prompts and rebuild the demos:

```bash
python3 scripts/download_model.py \
  --repo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out .local/models/qwen3-tts-voice-design
scripts/env.sh scripts/generate_speech.py \
  --model-path .local/models/qwen3-tts-voice-design \
  --samples examples/prompts.json --out outputs/demo-tts
scripts/env.sh -m speaking_midi outputs/demo-tts/en.wav --out outputs/demos/en
scripts/env.sh -m speaking_midi outputs/demo-tts/zh.wav --out outputs/demos/zh
scripts/env.sh -m scripts.package_examples
```

The generator resets its random seed before each sample and records the seed, generation mode, complete instruction, model revision, and measured duration. Model/library differences can change generated speech; use the committed `source.wav` files when comparing fitting methods on identical inputs. FluidSynth rerendering also has small numerical differences.
