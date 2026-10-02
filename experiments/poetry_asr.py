"""Unprompted source/v1/cached-direct ASR check; not an intelligibility claim."""
import json,re,time
from pathlib import Path
from importlib.metadata import version
from faster_whisper import WhisperModel
from experiments.backtest import fixtures

root=Path('outputs/poetry-backtest');cache=Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-base'
revision=(cache/'refs/main').read_text().strip()
model=WhisperModel(str(cache/'snapshots'/revision),device='cpu',compute_type='int8',cpu_threads=2)

def units(text,language):
    return [x for x in text if x.isalnum()] if language=='Chinese' else re.findall(r'[a-z0-9]+',text.lower())

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a):
        current=[i+1]
        for j,y in enumerate(b):current.append(min(current[-1]+1,row[j+1]+1,row[j]+(x!=y)))
        row=current
    return row[-1]

report=dict(model='Systran/faster-whisper-base',revision=revision,library=version('faster-whisper'),
            device='cpu',compute_type='int8',beam_size=5,vad_filter=False,condition_on_previous_text=False,
            prompt=None,results=[])
for sample in fixtures():
 for variant in ['source','v1','cached-direct']:
    path=root/sample['name']/(variant+'.wav')
    while not path.exists():time.sleep(2)
    segments,_=model.transcribe(str(path),language='zh' if sample['language']=='Chinese' else 'en',
                               beam_size=5,condition_on_previous_text=False,vad_filter=False)
    text=''.join(s.text for s in segments).strip();ref=units(sample['text'],sample['language'])
    row=dict(sample=sample['name'],variant=variant,hypothesis=text,reference=sample['text'],
             metric='CER' if sample['language']=='Chinese' else 'WER',
             error_rate=distance(ref,units(text,sample['language']))/len(ref))
    report['results'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    (root/'asr.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
