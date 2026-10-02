"""Unprompted source/v1/cached-direct ASR check; not an intelligibility claim."""
import argparse,hashlib,json,time
from pathlib import Path
from importlib.metadata import version
from faster_whisper import WhisperModel
from experiments.backtest import fixtures
from experiments.asr_scoring import units, distance

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('outputs/poetry-backtest'))
parser.add_argument('--variants',nargs='+',default=['source','v1','cached-direct'])
parser.add_argument('--out',type=Path)
parser.add_argument('--only',nargs='+')
parser.add_argument('--temperature',type=float,default=None)
args=parser.parse_args()
root=args.root;cache=Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-base'
revision=(cache/'refs/main').read_text().strip()
model=WhisperModel(str(cache/'snapshots'/revision),device='cpu',compute_type='int8',cpu_threads=2)

report=dict(model='Systran/faster-whisper-base',revision=revision,library=version('faster-whisper'),
            device='cpu',compute_type='int8',beam_size=5,vad_filter=False,condition_on_previous_text=False,
            prompt=None,temperature=args.temperature if args.temperature is not None else 'default fallback schedule',results=[])
for sample in fixtures():
 if args.only and sample['name'] not in args.only:continue
 for variant in args.variants:
    path=root/sample['name']/(variant+'.wav')
    while not path.exists():time.sleep(2)
    segments,_=model.transcribe(str(path),language='zh' if sample['language']=='Chinese' else 'en',
                               beam_size=5,condition_on_previous_text=False,vad_filter=False,
                               **({} if args.temperature is None else {'temperature':args.temperature}))
    text=''.join(s.text for s in segments).strip();ref=units(sample['text'],sample['language'])
    row=dict(sample=sample['name'],variant=variant,hypothesis=text,reference=sample['text'],
             audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             metric='CER' if sample['language']=='Chinese' else 'WER',
             error_rate=distance(ref,units(text,sample['language']))/len(ref))
    report['results'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    (args.out or root/'asr.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
