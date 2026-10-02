"""Explore balanced losses from the frozen quick MIDI initialization."""
import argparse
import hashlib
import json
from pathlib import Path
from speaking_midi.core import Piano, SR, load_audio, read_midi, render_saved
from speaking_midi.cached_refine import refine_cached
from speaking_midi.balanced_loss import BalancedLoss
from experiments.backtest import extra_metrics, fixtures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', default=['quiet-night', 'sonnet-18'])
    parser.add_argument('--weights', nargs='+', type=float, default=[.1, .2, .4])
    parser.add_argument('--device', default=None)
    parser.add_argument('--component', choices=['joint', 'spectral', 'envelope'], default='joint')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.out or (Path('outputs/balanced-search') if args.component == 'joint'
                        else Path('outputs/balanced-ablation')/args.component)
    piano = Piano()
    try:
        for sample in fixtures():
            if sample['name'] not in args.only:
                continue
            source = Path('outputs/poetry-backtest')/sample['name']
            y = load_audio(source/'source.wav')
            for weight in args.weights:
                out = root/sample['name']/f'w{weight:g}'
                out.mkdir(parents=True, exist_ok=True)
                spectral_weight = weight if args.component != 'envelope' else 0.
                envelope_weight = weight if args.component != 'spectral' else 0.
                objective = BalancedLoss(y, device=args.device, spectral_weight=spectral_weight,
                                         envelope_weight=envelope_weight)
                notes, trace = refine_cached(piano, y, read_midi(source/'quick.mid'),
                                            passes=4, objective=objective)
                audio = render_saved(piano, notes, out/'balanced', len(y)/SR+.5)
                components = {k: float(v[0]) for k, v in objective.components([audio]).items()}
                report = dict(extra_metrics(y, audio), envelope_nrmse=components['envelope_nrmse'],
                              objective=float(objective([audio])[0]), weight=weight, notes=len(notes),
                              component=args.component, spectral_weight=spectral_weight, envelope_weight=envelope_weight,
                              trace=trace, initial_midi_sha256=hashlib.sha256((source/'quick.mid').read_bytes()).hexdigest())
                (out/'report.json').write_text(json.dumps(report, indent=2))
                print('RESULT', sample['name'], weight, {k:v for k,v in report.items() if k!='trace'}, flush=True)
    finally:
        piano.close()

if __name__ == '__main__':
    main()
