"""Package measured fits as portable audio, plots, MIDI and a browser demo."""
import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import soundfile as sf

from speaking_midi.core import read_midi


def piano_roll(notes, seconds, path, language):
    fig, ax = plt.subplots(figsize=(12, 4))
    patches = [Rectangle((n.start, n.pitch - .4), n.duration, .8) for n in notes]
    collection = PatchCollection(patches, cmap='viridis', edgecolor='none')
    collection.set_array([n.velocity for n in notes])
    collection.set_clim(1, 127)
    ax.add_collection(collection)
    low = min(n.pitch for n in notes) - 2
    high = max(n.pitch for n in notes) + 2
    ax.set(xlim=(0, seconds), ylim=(low, high), xlabel='Time (seconds)', ylabel='MIDI pitch',
           title=f'{language}: exported piano MIDI (color = velocity)')
    ax.set_yticks(range((low // 12 + 1) * 12, high + 1, 12))
    ax.grid(alpha=.2)
    fig.colorbar(collection, ax=ax, label='Velocity')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tts', type=Path, default=Path('outputs/demo-tts'))
    parser.add_argument('--fits', type=Path, default=Path('outputs/demos'))
    parser.add_argument('--out', type=Path, default=Path('examples'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.tts / 'manifest.json').read_text())
    instructions = {row.get('instruct') for row in manifest['samples']}
    if manifest.get('mode') != 'voice_design' or len(instructions) != 1 or not next(iter(instructions)):
        raise ValueError('These demos require VoiceDesign samples sharing one nonempty instruction')
    instruction = next(iter(instructions))
    # Keep the model identity, not the machine-specific path used to load it.
    manifest.pop('snapshot', None)
    manifest['fitting'] = dict(max_notes=240, renderer='FluidSynth / bank 0 / program 0',
                               soundfont='gm.sf3 / Grand Piano')
    manifest['assets'] = {}
    cards = []
    for row in manifest['samples']:
        name = row['name']
        fit = args.fits / name
        dest = args.out / name
        dest.mkdir(exist_ok=True)
        for stem in ['baseline', 'dictionary', 'refined']:
            for suffix in ['mid', 'wav', 'json']:
                shutil.copyfile(fit / f'{stem}.{suffix}', dest / f'{stem}.{suffix}')
        for filename in ['target.wav', 'spectrogram.png']:
            shutil.copyfile(fit / filename, dest / filename)
        source_audio = args.tts / f'{name}.wav'
        if not source_audio.exists():
            source_audio = args.tts / name / 'source.wav'
        if source_audio.resolve() != (dest / 'source.wav').resolve():
            shutil.copyfile(source_audio, dest / 'source.wav')
        metrics = json.loads((fit / 'metrics.json').read_text())
        metrics['input'] = f'examples/{name}/source.wav'
        metrics['caveats'][0] = 'These longer demo samples have no human intelligibility or ASR evaluation.'
        (dest / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        notes = read_midi(dest / 'refined.mid')
        if not notes:
            raise ValueError(f'No notes in {name}')
        seconds = sf.info(dest / 'refined.wav').duration
        piano_roll(notes, seconds, dest / 'piano-roll.png', row['language'])
        title = html.escape(row['language'])
        transcript = html.escape(row['text'])
        players = ''.join(
            f'<label>{label}<audio controls preload="metadata" src="{name}/{stem}.wav"'
            f'{" data-piano=" + name if stem == "refined" else ""}></audio></label>'
            for stem, label in [('target', 'Target speech (common RMS)'),
                                ('baseline', 'Spectral baseline'), ('refined', 'Fitted piano')])
        baseline = metrics['results']['baseline']['log_band_mae']
        refined = metrics['results']['refined']['log_band_mae']
        improvement = 100 * (baseline - refined) / baseline
        cards.append(f'''<section>
<h2>{title} · {row['seconds']:.2f} s</h2>
<p>{len(notes)} MIDI notes · log-band error {baseline:.4f} → {refined:.4f} ({improvement:.1f}% reduction).</p>
<div class="players">{players}</div>
<details><summary>Reveal transcript</summary><p lang="{name}">{transcript}</p></details>
<p><a href="{name}/refined.mid">Download MIDI</a> · <a href="{name}/source.wav">Original Qwen output</a> · <a href="{name}/metrics.json">Metrics</a></p>
<h3>MIDI piano roll</h3><p>Play the fitted piano to follow the cursor. Click the roll to seek. Colors show velocity; rectangles show key-hold duration, not release tails.</p>
<p class="velocity-key">Velocity: 1 <span aria-hidden="true"></span> 127</p>
<canvas data-roll="{name}" role="img" aria-label="{title} MIDI piano roll"></canvas>
<noscript><img src="{name}/piano-roll.png" alt="{title} piano roll"></noscript>
<script type="application/json" id="notes-{name}">{json.dumps(dict(seconds=seconds, notes=[asdict(n) for n in notes]))}</script>
<h3>Shared-scale spectrogram comparison</h3><img src="{name}/spectrogram.png" alt="{title}: target, baseline, dictionary and refined spectrograms">
</section>''')
        manifest['assets'][name] = {
            p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(dest.iterdir()) if p.is_file()
        }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    template = Path(__file__).with_name('demo_template.html').read_text()
    (args.out / 'index.html').write_text(template.replace('<!-- DEMO_CARDS -->', '\n'.join(cards)).replace('<!-- VOICE_INSTRUCTION -->', html.escape(instruction)))
    print(f'Packaged {len(cards)} demos in {args.out}', flush=True)


if __name__ == '__main__':
    main()
