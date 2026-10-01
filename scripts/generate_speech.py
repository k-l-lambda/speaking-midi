"""Generate speech fixtures with Qwen3-TTS and record their provenance."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path')
    parser.add_argument('--model', help='Hugging Face model ID; inferred from local source.json when available')
    parser.add_argument('--samples', type=Path, help='JSON list of name/text/language and speaker or instruct records')
    parser.add_argument('--out', type=Path, default=Path('outputs/tts'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only', help='Generate only the named sample')
    parser.add_argument('--instruct', help='Override the voice description for every sample')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.model_path) / 'source.json' if args.model_path else None
    source = json.loads(source_path.read_text()) if source_path and source_path.exists() else None
    repo = args.model or (source or {}).get('repo') or 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
    if source and args.model and source['repo'] != args.model:
        raise ValueError('Requested model does not match local model provenance')
    path = args.model_path or snapshot_download(repo)
    rows = json.loads(args.samples.read_text()) if args.samples else [
        dict(name='zh', text='你好，今天的天气真好。', language='Chinese', speaker='Vivian'),
        dict(name='en', text='Hello, can you hear the piano speaking?', language='English', speaker='Ryan'),
    ]
    if args.only:
        rows = [row for row in rows if row['name'] == args.only]
    names = [row['name'] for row in rows]
    if not names or len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
        raise ValueError('Samples must have unique file stems, and the selection must not be empty')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Qwen3TTSModel.from_pretrained(
        path, device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    mode = model.model.tts_model_type
    samples = []
    for row in rows:
        instruction = args.instruct if args.instruct is not None else row.get('instruct')
        seed = row.get('seed', args.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        start = time.time()
        if mode == 'voice_design':
            if not instruction:
                raise ValueError('VoiceDesign requires an instruct field or --instruct')
            wavs, sr = model.generate_voice_design(
                text=row['text'], language=row['language'], instruct=instruction, max_new_tokens=512)
        elif mode == 'custom_voice':
            if instruction and model.model.tts_model_size == '0b6':
                raise ValueError('0.6B CustomVoice ignores instructions; use a supported 1.7B model')
            wavs, sr = model.generate_custom_voice(
                text=row['text'], language=row['language'], speaker=row['speaker'],
                instruct=instruction, max_new_tokens=512)
        else:
            raise ValueError(f'Unsupported generation mode: {mode}')
        sf.write(args.out / f"{row['name']}.wav", wavs[0], sr, subtype='FLOAT')
        record = dict(row, seed=seed, sample_rate=sr, seconds=len(wavs[0]) / sr,
                      wall_seconds=time.time() - start)
        if instruction:
            record['instruct'] = instruction
        if mode == 'voice_design':
            record.pop('speaker', None)
        samples.append(record)
        print(record, flush=True)
    manifest = dict(model=repo, snapshot=str(path), source=source, mode=mode,
                    seed=args.seed, seed_policy='Reset RNG before each sample',
                    dtype='bfloat16', attention='sdpa', max_new_tokens=512, samples=samples)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
