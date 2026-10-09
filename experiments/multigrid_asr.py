"""Score actual saved-MIDI renders without exporting WAV or prompting the ASR."""
import json
from pathlib import Path
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from speaking_midi.core import Piano, SR, load_audio, read_midi
from speaking_midi.dense import fit_dense
from speaking_midi.instrumental import save_midi
from experiments.chime_objective import measure
from experiments.asr_scoring import score


def main():
    cache = Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-base'
    revision = (cache/'refs/main').read_text().strip()
    model = WhisperModel(str(cache/'snapshots'/revision), device='cpu', compute_type='int8')
    reference = '床前明月光，疑是地上霜。举头望明月，低头思故乡。'
    y = load_audio('examples/poetry/quiet-night.wav', target_rms=None)[:round(8.64*SR)]
    results = []
    p = Piano()
    try:
        dense_path = Path('outputs/multigrid/quiet-night-dense.mid')
        if not dense_path.exists():
            notes, report, _ = fit_dense(y, p, steps=600)
            save_midi(notes, dense_path, len(y)/SR)
            saved = read_midi(dense_path)
            report['evaluation'] = measure(y, p.render(saved, len(y)/SR+.5), saved)
            dense_path.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
        for name in ['source', 'legacy', 'suppressed', 'multigrid', 'multiwindow', 'dense']:
            wave = y if name == 'source' else p.render(
                read_midi(f'outputs/multigrid/quiet-night-{name}.mid'), len(y)/SR+.5)
            segments, _ = model.transcribe(resample_poly(wave, 2, 3), language='zh',
                beam_size=5, temperature=0, vad_filter=False, condition_on_previous_text=False)
            text = ''.join(s.text for s in segments)
            results.append(dict(variant=name, text=text, **score(reference, text, 'Chinese')))
    finally:
        p.close()
    output = dict(model='Systran/faster-whisper-base', revision=revision,
                  reference=reference, results=results)
    Path('outputs/multigrid/asr.json').write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
