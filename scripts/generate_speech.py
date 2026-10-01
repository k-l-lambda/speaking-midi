import argparse, json, pathlib, time
import numpy as np
import torch
import soundfile as sf
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

out = pathlib.Path('outputs/tts'); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
repo = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
parser=argparse.ArgumentParser(); parser.add_argument('--model-path'); args=parser.parse_args()
path = args.model_path or snapshot_download(repo)
model = Qwen3TTSModel.from_pretrained(path, device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
rows = [('zh', '你好，今天的天气真好。', 'Chinese', 'Vivian'),
        ('en', 'Hello, can you hear the piano speaking?', 'English', 'Ryan')]
meta = []
for name, text, language, speaker in rows:
    start = time.time()
    wavs, sr = model.generate_custom_voice(text=text, language=language, speaker=speaker, max_new_tokens=512)
    sf.write(out / f'{name}.wav', wavs[0], sr, subtype='FLOAT')
    meta.append(dict(name=name, text=text, language=language, speaker=speaker, sample_rate=sr,
                     seconds=len(wavs[0])/sr, wall_seconds=time.time()-start))
    print(meta[-1], flush=True)
(out/'manifest.json').write_text(json.dumps(dict(model=repo, snapshot=path, source=json.loads((pathlib.Path(path)/'source.json').read_text()) if (pathlib.Path(path)/'source.json').exists() else None, seed=42, samples=meta), ensure_ascii=False, indent=2))
